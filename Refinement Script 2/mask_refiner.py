import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology, measure, filters
from PIL import Image
import os
import argparse
import time
from typing import Union, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

warnings.filterwarnings('ignore')


class MaskRefiner:
    """
    A class for refining segmentation masks with multiple post-processing techniques.
    Optimized for bladder segmentation masks but can be used for other organs.
    """
    def __init__(self, 
                 min_object_size: int = 100,
                 hole_fill_area_threshold: int = 50,
                 morphology_kernel_size: int = 3,
                 gaussian_sigma: float = 0.5,
                 use_adaptive_threshold: bool = True,
                 preserve_largest_component: bool = True):
        """
        Initialize the MaskRefiner with configurable parameters.
        
        Args:
            min_object_size: Minimum size for objects to keep (removes small noise)
            hole_fill_area_threshold: Maximum hole size to fill
            morphology_kernel_size: Size of morphological operations kernel
            gaussian_sigma: Sigma for Gaussian smoothing
            use_adaptive_threshold: Whether to use adaptive thresholding
            preserve_largest_component: Whether to keep only the largest connected component
        """
        self.min_object_size = min_object_size
        self.hole_fill_area_threshold = hole_fill_area_threshold
        self.morphology_kernel_size = morphology_kernel_size
        self.gaussian_sigma = gaussian_sigma
        self.use_adaptive_threshold = use_adaptive_threshold
        self.preserve_largest_component = preserve_largest_component
        
        # Create morphological kernels
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morphology_kernel_size, morphology_kernel_size)
        )
        
    def adaptive_threshold(self, probability_map: np.ndarray) -> float:
        """
        Compute adaptive threshold using Otsu's method on probability map.
        
        Args:
            probability_map: Input probability map (0-1 range)
            
        Returns:
            Optimal threshold value
        """
        prob_8bit = (probability_map * 255).astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise before thresholding
        prob_8bit = cv2.GaussianBlur(prob_8bit, (5, 5), 0)
        
        # Compute Otsu threshold
        threshold_val, _ = cv2.threshold(prob_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return threshold_val / 255.0
    
    def remove_small_objects(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Remove small connected components from binary mask.
        
        Args:
            binary_mask: Binary mask (0 or 255)
            
        Returns:
            Cleaned binary mask
        """
        # Convert to boolean for skimage
        mask_bool = binary_mask > 0
        
        # Remove small objects
        cleaned = morphology.remove_small_objects(
            mask_bool, 
            min_size=self.min_object_size,
            connectivity=2
        )
        
        return (cleaned * 255).astype(np.uint8)
    
    def fill_holes(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in binary mask with size-based filtering.
        
        Args:
            binary_mask: Binary mask (0 or 255)
            
        Returns:
            Mask with holes filled
        """
        # Convert to boolean
        mask_bool = binary_mask > 0
        
        # Fill holes
        filled = ndimage.binary_fill_holes(mask_bool)
        
        # If we want to be more selective about hole filling
        if self.hole_fill_area_threshold > 0:
            # Find holes and filter by size
            holes = filled & ~mask_bool
            labeled_holes = measure.label(holes)
            
            for region in measure.regionprops(labeled_holes):
                if region.area > self.hole_fill_area_threshold:
                    # Don't fill large holes
                    filled[labeled_holes == region.label] = False
        
        return (filled * 255).astype(np.uint8)
    
    def morphological_operations(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening and closing to smooth contours.
        
        Args:
            binary_mask: Binary mask (0 or 255)
            
        Returns:
            Smoothed binary mask
        """
        # Opening: erosion followed by dilation (removes noise, smooths contours)
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Closing: dilation followed by erosion (fills small holes, connects nearby objects)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.kernel)
        
        return closed
    
    def get_largest_component(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Keep only the largest connected component.
        
        Args:
            binary_mask: Binary mask (0 or 255)
            
        Returns:
            Mask with only largest component
        """
        # Find connected components
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        if num_labels <= 1:  # No objects found
            return binary_mask
        
        # Find largest component (excluding background label 0)
        largest_label = 1
        largest_size = 0
        
        for label in range(1, num_labels):
            size = np.sum(labels == label)
            if size > largest_size:
                largest_size = size
                largest_label = label
        
        # Create mask with only largest component
        largest_component = (labels == largest_label).astype(np.uint8) * 255
        
        return largest_component
    
    def gaussian_smoothing(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to reduce jagged edges.
        
        Args:
            binary_mask: Binary mask (0 or 255)
            
        Returns:
            Smoothed binary mask
        """
        if self.gaussian_sigma <= 0:
            return binary_mask
            
        # Convert to float for smoothing
        mask_float = binary_mask.astype(np.float32) / 255.0
        
        # Apply Gaussian filter
        smoothed = filters.gaussian(mask_float, sigma=self.gaussian_sigma)
        
        # Threshold back to binary
        smoothed_binary = (smoothed > 0.5).astype(np.uint8) * 255
        
        return smoothed_binary
    
    def smooth_contour(self, binary_mask: np.ndarray, smoothing_factor: float = 0.1) -> np.ndarray:
        """
        Smooth the contour of the mask using spline interpolation.

        Args:
            binary_mask: The binary mask (0 or 255).
            smoothing_factor: A factor to control the amount of smoothing.
                              s=0 -> no smoothing (interpolates through all points)
                              Larger s -> more smoothing.

        Returns:
            A new binary mask with a smoothed contour.
        """
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return binary_mask

        # Assuming the largest contour is the one we want to smooth
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 4:  # Spline fitting requires at least 4 points
            return binary_mask

        # The contour is a list of points (x,y). We need to unpack it for splprep.
        x, y = largest_contour.squeeze().T

        # The points need to be in order, which they are from findContours.
        # We also need to close the loop for a closed shape.
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

        # Create a B-spline representation of the contour.
        # s is the smoothing factor. A larger s means more smoothing.
        tck, u = splprep([x, y], s=smoothing_factor * len(x), per=True)

        # Evaluate the spline at a higher resolution to get a smooth curve.
        unew = np.linspace(0, 1, 1000)
        out = splev(unew, tck)

        # Create a new mask from the smoothed contour
        smoothed_mask = np.zeros_like(binary_mask)
        smoothed_contour = np.array([out]).T.reshape((-1, 1, 2)).astype(np.int32)

        # Draw the smoothed contour and fill it
        cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 255, thickness=cv2.FILLED)

        return smoothed_mask

    def refine_mask(self, 
                   input_mask: Union[np.ndarray, str], 
                   probability_map: Optional[np.ndarray] = None,
                   custom_threshold: Optional[float] = None,
                   smoothing_factor: Optional[float] = None) -> np.ndarray:
        """
        Main function to refine a segmentation mask.
        
        Args:
            input_mask: Either a file path to mask image or numpy array
            probability_map: Optional probability map for adaptive thresholding
            custom_threshold: Custom threshold value (overrides adaptive)
            smoothing_factor: Optional factor for contour smoothing.
            
        Returns:
            Refined binary mask
        """
        # Load mask if it's a file path
        if isinstance(input_mask, str):
            mask = np.array(Image.open(input_mask).convert('L'))
        else:
            mask = input_mask.copy()
        
        # Handle probability maps vs binary masks
        if probability_map is not None:
            # Use probability map for thresholding
            if custom_threshold is not None:
                threshold = custom_threshold
            elif self.use_adaptive_threshold:
                threshold = self.adaptive_threshold(probability_map)
            else:
                threshold = 0.5
            
            mask = (probability_map > threshold).astype(np.uint8) * 255
        else:
            # Assume input is already binary or make it binary
            if mask.max() <= 1:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = (mask > 127).astype(np.uint8) * 255
        
        # Step 1: Remove small objects
        mask = self.remove_small_objects(mask)
        
        # Step 2: Fill holes
        mask = self.fill_holes(mask)
        
        # Step 3: Morphological operations for smoothing
        mask = self.morphological_operations(mask)
        
        # Step 4: Keep largest component if requested
        if self.preserve_largest_component:
            mask = self.get_largest_component(mask)
        
        # Step 5: Final smoothing
        mask = self.gaussian_smoothing(mask)

        # Step 6: Contour smoothing if requested
        if smoothing_factor is not None and smoothing_factor > 0:
            mask = self.smooth_contour(mask, smoothing_factor)
        
        return mask
    
    def batch_refine(self, 
                    input_dir: str, 
                    output_dir: str,
                    suffix: str = "_refined",
                    verbose: bool = True,
                    smoothing_factor: Optional[float] = None) -> None:
        """
        Batch process multiple mask files.
        
        Args:
            input_dir: Directory containing input masks
            output_dir: Directory to save refined masks
            suffix: Suffix to add to output filenames
            verbose: Whether to print progress
            smoothing_factor: Optional factor for contour smoothing.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        mask_files = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        if verbose:
            print(f"Found {len(mask_files)} mask files to process")
        
        total_time = 0
        
        for i, filename in enumerate(mask_files):
            start_time = time.time()
            
            input_path = os.path.join(input_dir, filename)
            
            # Create output filename
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}{suffix}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # Refine mask
                refined_mask = self.refine_mask(input_path, smoothing_factor=smoothing_factor)
                
                # Save refined mask
                Image.fromarray(refined_mask).save(output_path)
                
                process_time = time.time() - start_time
                total_time += process_time
                
                if verbose:
                    print(f"[{i+1}/{len(mask_files)}] Processed {filename} -> {output_filename} "
                          f"({process_time:.3f}s)")
                    
            except Exception as e:
                if verbose:
                    print(f"Error processing {filename}: {str(e)}")
        
        if verbose:
            avg_time = total_time / len(mask_files) if mask_files else 0
            print(f"\nBatch processing complete!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per mask: {avg_time:.3f}s")


def create_comparison_images(original_dir: str,
                           refined_dir: str,
                           comparison_dir: str,
                           sample_files: Optional[list] = None) -> None:
    """
    Create side-by-side comparison images for each original vs. refined mask pair.

    Args:
        original_dir: Directory with original masks.
        refined_dir: Directory with refined masks.
        comparison_dir: Directory to save comparison images.
        sample_files: Specific files to compare (optional).
    """
    os.makedirs(comparison_dir, exist_ok=True)

    if sample_files is None:
        # Get all matching files
        orig_files = set(os.listdir(original_dir))
        refined_files = set(os.listdir(refined_dir))

        # Find files with refined suffix
        matching_files = []
        for ref_file in refined_files:
            if ref_file.endswith('_refined.png'):
                orig_name = ref_file.replace('_refined.png', '.png')
                if orig_name in orig_files:
                    matching_files.append((orig_name, ref_file))
        
        sample_files = matching_files

    if not sample_files:
        print("No matching files found for comparison")
        return

    print(f"Generating {len(sample_files)} comparison images in '{comparison_dir}'...")

    for i, (orig_file, refined_file) in enumerate(sample_files):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Load images
        orig_mask = np.array(Image.open(os.path.join(original_dir, orig_file)).convert('L'))
        refined_mask = np.array(Image.open(os.path.join(refined_dir, refined_file)).convert('L'))

        # Plot original
        axes[0].imshow(orig_mask, cmap='gray')
        axes[0].set_title(f'Original: {orig_file}')
        axes[0].axis('off')

        # Plot refined
        axes[1].imshow(refined_mask, cmap='gray')
        axes[1].set_title(f'Refined: {refined_file}')
        axes[1].axis('off')

        plt.tight_layout()

        # Save the individual comparison image
        base_name = os.path.splitext(orig_file)[0]
        output_path = os.path.join(comparison_dir, f"{base_name}_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

    print(f"Successfully generated {len(sample_files)} comparison images.")


def create_diff_image(original_mask_path: str, refined_mask_path: str, diff_output_path: str):
    """
    Creates a visual difference image between two masks.

    - White: Pixels are the same in both masks.
    - Green: Pixels were added in the refined mask.
    - Red:   Pixels were removed in the refined mask.
    """
    original = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
    refined = cv2.imread(refined_mask_path, cv2.IMREAD_GRAYSCALE)

    if original is None or refined is None:
        print(f"Error: Could not read one of the masks for diff.")
        return

    # Create a 3-channel BGR image for color differences
    diff_image = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)

    # Where pixels are the same (both background or both foreground)
    same_bg = (original == 0) & (refined == 0)
    same_fg = (original > 0) & (refined > 0)
    diff_image[same_bg] = [0, 0, 0]       # Black for same background
    diff_image[same_fg] = [255, 255, 255] # White for same foreground

    # Where pixels were added (orig is background, refined is foreground)
    added = (original == 0) & (refined > 0)
    diff_image[added] = [0, 255, 0]  # Green for added pixels

    # Where pixels were removed (orig is foreground, refined is background)
    removed = (original > 0) & (refined == 0)
    diff_image[removed] = [0, 0, 255]  # Red for removed pixels

    cv2.imwrite(diff_output_path, diff_image)
    print(f"Difference image saved to: {diff_output_path}")


def create_trimap(mask_path: str, output_path: str, border_size: int = 5):
    """
    Creates a trimap from a binary mask.

    The trimap has three regions:
    - 255 (Sure Foreground): Eroded mask.
    - 128 (Unknown Boundary): Region between eroded and dilated mask.
    - 0 (Sure Background): Everything else.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask for trimap: {mask_path}")
        return

    kernel = np.ones((border_size, border_size), np.uint8)

    # Erode the mask to get the "sure foreground"
    sure_fg = cv2.erode(mask, kernel)

    # Dilate the mask to define the "unknown" region
    dilated_mask = cv2.dilate(mask, kernel)

    # Create the trimap
    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[dilated_mask > 0] = 128  # Unknown region
    trimap[sure_fg > 0] = 255    # Sure foreground

    cv2.imwrite(output_path, trimap)
    print(f"Trimap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Refine segmentation masks')
    parser.add_argument('--input_dir', default='segmentation_masks',
                       help='Directory containing input masks')
    parser.add_argument('--output_dir', default='refined_masks',
                       help='Directory to save refined masks')
    parser.add_argument('--min_object_size', type=int, default=100,
                       help='Minimum object size to keep')
    parser.add_argument('--hole_fill_threshold', type=int, default=50,
                       help='Maximum hole size to fill')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Morphological kernel size')
    parser.add_argument('--gaussian_sigma', type=float, default=0.5,
                       help='Gaussian smoothing sigma')
    parser.add_argument('--preserve_largest', action='store_true',
                       help='Keep only largest connected component')
    parser.add_argument('--create_comparison', action='store_true',
                       help='Create before/after comparison images for each mask')
    parser.add_argument('--comparison_dir', default='comparison',
                        help='Directory to save comparison images')
    parser.add_argument('--create_diff', action='store_true',
                        help='Create a pixel-wise difference image for each mask.')
    parser.add_argument('--generate_trimap', action='store_true',
                        help='Generate a trimap for each mask for boundary-aware training.')
    parser.add_argument('--trimap_border_size', type=int, default=5,
                        help='The pixel width of the "unknown" border in the trimap.')
    parser.add_argument('--smooth_contour', type=float, default=None,
                        help='Factor for contour spline smoothing. Try values like 0.5 or 1.0. Applied at the end.')
    parser.add_argument('--single_file', type=str,
                       help='Process single file instead of batch')

    args = parser.parse_args()

    # Initialize refiner
    refiner = MaskRefiner(
        min_object_size=args.min_object_size,
        hole_fill_area_threshold=args.hole_fill_threshold,
        morphology_kernel_size=args.kernel_size,
        gaussian_sigma=args.gaussian_sigma,
        preserve_largest_component=args.preserve_largest
    )

    # Determine if we are just generating images or processing
    is_image_generation_only = args.create_comparison or args.create_diff or args.generate_trimap

    if args.single_file:
        if not is_image_generation_only:
            # Process single file
            print(f"Processing single file: {args.single_file}")
            refined_mask = refiner.refine_mask(
                args.single_file,
                smoothing_factor=args.smooth_contour
            )
            # Save result
            output_name = os.path.splitext(os.path.basename(args.single_file))[0] + "_refined.png"
            output_path = os.path.join(args.output_dir, output_name)
            os.makedirs(args.output_dir, exist_ok=True)
            Image.fromarray(refined_mask).save(output_path)
            print(f"Refined mask saved to: {output_path}")

    else: # Batch processing
        if not is_image_generation_only:
            print(f"Batch processing masks from {args.input_dir}")
            refiner.batch_refine(
                args.input_dir,
                args.output_dir,
                smoothing_factor=args.smooth_contour
            )

    # Handle image generation tasks separately
    if is_image_generation_only:
        print("\n--- Image Generation ---")
        input_dir = os.path.dirname(args.single_file) if args.single_file else args.input_dir
        
        # Get all original mask files
        mask_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
        if args.single_file:
            mask_files = [os.path.basename(args.single_file)]

        for filename in mask_files:
            original_path = os.path.join(input_dir, filename)
            refined_name = os.path.splitext(filename)[0] + "_refined.png"
            refined_path = os.path.join(args.output_dir, refined_name)

            if not os.path.exists(refined_path):
                print(f"Skipping {filename}, no refined mask found.")
                continue

            if args.create_comparison:
                os.makedirs(args.comparison_dir, exist_ok=True)
                comp_name = os.path.splitext(filename)[0] + "_comparison.png"
                comp_path = os.path.join(args.comparison_dir, comp_name)
                create_comparison_images(
                    original_dir=input_dir,
                    refined_dir=args.output_dir,
                    comparison_dir=args.comparison_dir,
                    sample_files=[(filename, refined_name)]
                )
            
            if args.create_diff:
                os.makedirs(args.output_dir, exist_ok=True) # Ensure output dir exists
                diff_name = os.path.splitext(filename)[0] + "_diff.png"
                diff_path = os.path.join(args.output_dir, diff_name)
                create_diff_image(original_path, refined_path, diff_path)
            
            if args.generate_trimap:
                # We typically generate the trimap from the *refined* mask
                # as it's cleaner.
                os.makedirs(args.output_dir, exist_ok=True)
                trimap_name = os.path.splitext(filename)[0] + "_trimap.png"
                trimap_path = os.path.join(args.output_dir, trimap_name)
                create_trimap(refined_path, trimap_path, args.trimap_border_size)


if __name__ == "__main__":
    main() 