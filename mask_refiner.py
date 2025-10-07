"""
Unified Mask Refinement Pipeline for Ultrasound Segmentation
Conservative approach to avoid over-smoothing while improving contour quality
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage import morphology, measure
import os
from typing import Tuple, Optional, Dict
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class UnifiedMaskRefiner:
    """
    Conservative mask refinement for ultrasound segmentation.
    Combines morphological, spline, and Fourier-based smoothing with safety checks.
    """

    def __init__(self,
                 min_area_change: float = 0.05,  # Max 5% area change allowed
                 max_hausdorff_increase: float = 8.0,  # Max 8 pixel increase in HD for actual smoothing
                 smoothing_strength: str = 'moderate'):  # 'conservative', 'moderate', 'aggressive'
        """
        Initialize refiner with safety thresholds to prevent over-smoothing.

        Args:
            min_area_change: Maximum allowed relative area change (default 2%)
            max_hausdorff_increase: Maximum allowed Hausdorff distance increase in pixels
            smoothing_strength: Preset strength level for smoothing operations
        """
        self.min_area_change = min_area_change
        self.max_hausdorff_increase = max_hausdorff_increase

        # Set parameters based on smoothing strength
        self.params = self._get_params_by_strength(smoothing_strength)
        self.strength = smoothing_strength

    def _get_params_by_strength(self, strength: str) -> Dict:
        """Get parameter presets based on desired smoothing strength."""
        presets = {
            'conservative': {
                'morph_kernel': 3,
                'hole_threshold': 30,
                'min_object_size': 100,
                'spline_smooth': 0.05,
                'fourier_keep': 40,
                'gaussian_sigma': 0.3
            },
            'moderate': {
                'morph_kernel': 5,
                'hole_threshold': 50,
                'min_object_size': 150,
                'spline_smooth': 0.3,
                'fourier_keep': 25,
                'gaussian_sigma': 1.0
            },
            'aggressive': {
                'morph_kernel': 7,
                'hole_threshold': 75,
                'min_object_size': 200,
                'spline_smooth': 0.2,
                'fourier_keep': 20,
                'gaussian_sigma': 0.8
            }
        }
        return presets.get(strength, presets['conservative'])

    def _ensure_binary(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask is binary (0 or 255)."""
        return ((mask > 127).astype(np.uint8) * 255)

    def _calculate_metrics(self, original: np.ndarray, refined: np.ndarray) -> Dict:
        """Calculate quality metrics between original and refined masks."""
        # Area change
        orig_area = np.sum(original > 0)
        ref_area = np.sum(refined > 0)
        area_change = abs(ref_area - orig_area) / (orig_area + 1e-6)

        # Hausdorff distance (simplified version using contour distance)
        orig_contours, _ = cv2.findContours(original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ref_contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        hausdorff = 0
        if orig_contours and ref_contours:
            orig_pts = orig_contours[0].reshape(-1, 2)
            ref_pts = ref_contours[0].reshape(-1, 2)

            # Sample points for faster computation
            sample_size = min(100, len(orig_pts), len(ref_pts))
            orig_sample = orig_pts[::max(1, len(orig_pts)//sample_size)]
            ref_sample = ref_pts[::max(1, len(ref_pts)//sample_size)]

            # Compute directed Hausdorff distances
            if len(orig_sample) > 0 and len(ref_sample) > 0:
                dist1 = np.max([np.min(np.linalg.norm(ref_sample - p, axis=1)) for p in orig_sample])
                dist2 = np.max([np.min(np.linalg.norm(orig_sample - p, axis=1)) for p in ref_sample])
                hausdorff = max(dist1, dist2)

        return {
            'area_change': area_change,
            'hausdorff': hausdorff,
            'orig_area': orig_area,
            'ref_area': ref_area
        }

    def _morphological_smoothing(self, mask: np.ndarray) -> np.ndarray:
        """Apply gentle morphological operations."""
        kernel_size = self.params['morph_kernel']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Fill small holes first
        filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Remove small protrusions
        smoothed = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel, iterations=1)

        return smoothed

    def _remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected components."""
        min_size = self.params['min_object_size']

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Keep only components larger than threshold
        result = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                result[labels == i] = 255

        return result

    def _spline_smooth_contour(self, mask: np.ndarray) -> np.ndarray:
        """Apply spline smoothing to the largest contour."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return mask

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) < 10:  # Need enough points for spline
            return mask

        # Extract points
        points = largest_contour.squeeze()
        if points.ndim != 2:
            return mask

        x, y = points[:, 0], points[:, 1]

        # Close the contour
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        try:
            # Fit spline with conservative smoothing
            smooth_factor = self.params['spline_smooth'] * len(x)
            tck, u = splprep([x, y], s=smooth_factor, per=True, k=3)

            # Evaluate spline
            u_new = np.linspace(0, 1, len(x))
            x_new, y_new = splev(u_new, tck)

            # Create new mask
            smoothed_contour = np.column_stack([x_new, y_new]).astype(np.int32)
            result = np.zeros_like(mask)
            cv2.fillPoly(result, [smoothed_contour], 255)

            return result
        except:
            # If spline fails, return original
            return mask

    def _fourier_smooth(self, mask: np.ndarray) -> np.ndarray:
        """Apply Fourier descriptor smoothing."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return mask

        largest_contour = max(contours, key=cv2.contourArea)
        points = largest_contour.squeeze()

        if points.ndim != 2 or len(points) < 10:
            return mask

        # Fourier transform of contour
        complex_contour = points[:, 0] + 1j * points[:, 1]
        fourier_coeff = np.fft.fft(complex_contour)

        # Keep only low-frequency components
        keep = self.params['fourier_keep']
        if keep < len(fourier_coeff) // 2:
            fourier_coeff[keep:-keep] = 0

        # Inverse transform
        smoothed_complex = np.fft.ifft(fourier_coeff)
        smoothed_points = np.column_stack([
            smoothed_complex.real,
            smoothed_complex.imag
        ]).astype(np.int32)

        # Create new mask
        result = np.zeros_like(mask)
        cv2.fillPoly(result, [smoothed_points], 255)

        return result

    def refine(self, mask: np.ndarray,
               method: str = 'hybrid',
               check_quality: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Refine a mask using specified method with quality checks.

        Args:
            mask: Input binary mask
            method: 'morphological', 'spline', 'fourier', or 'hybrid'
            check_quality: Whether to enforce quality constraints

        Returns:
            refined_mask: The refined mask
            metrics: Dictionary of quality metrics
        """
        # Ensure binary
        mask = self._ensure_binary(mask)
        original = mask.copy()

        # Apply refinement based on method
        if method == 'morphological':
            refined = self._morphological_smoothing(mask)
            refined = self._remove_small_components(refined)

        elif method == 'spline':
            refined = self._remove_small_components(mask)
            refined = self._spline_smooth_contour(refined)

        elif method == 'fourier':
            refined = self._remove_small_components(mask)
            refined = self._fourier_smooth(refined)

        elif method == 'hybrid':
            # Progressive refinement: morph -> spline -> light fourier
            refined = self._morphological_smoothing(mask)
            refined = self._remove_small_components(refined)

            # Check intermediate quality
            metrics = self._calculate_metrics(original, refined)
            if metrics['area_change'] < self.min_area_change:
                refined = self._spline_smooth_contour(refined)

                # Check again
                metrics = self._calculate_metrics(original, refined)
                if metrics['area_change'] < self.min_area_change and \
                   metrics['hausdorff'] < self.max_hausdorff_increase:
                    # Apply very light Fourier smoothing
                    temp_params = self.params.copy()
                    self.params['fourier_keep'] = 50  # Keep more coefficients
                    refined = self._fourier_smooth(refined)
                    self.params = temp_params
        else:
            refined = mask

        # Final quality check
        metrics = self._calculate_metrics(original, refined)

        if check_quality:
            # Revert if quality degraded
            if metrics['area_change'] > self.min_area_change or \
               metrics['hausdorff'] > self.max_hausdorff_increase:
                print(f"Warning: Refinement degraded quality. Reverting.")
                print(f"  Area change: {metrics['area_change']:.3f}")
                print(f"  Hausdorff increase: {metrics['hausdorff']:.1f}")
                refined = original
                metrics = self._calculate_metrics(original, refined)

        return refined, metrics

    def create_difference_visualization(self, original: np.ndarray,
                                       refined: np.ndarray,
                                       image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization showing differences between original and refined masks.

        Green: Added pixels
        Red: Removed pixels
        White: Unchanged mask pixels
        Gray: Background

        Args:
            original: Original mask
            refined: Refined mask
            image: Optional ultrasound image for background

        Returns:
            visualization: Color image showing differences
        """
        original = self._ensure_binary(original)
        refined = self._ensure_binary(refined)

        # Create base visualization
        if image is not None and len(image.shape) == 2:
            # Convert grayscale ultrasound to RGB
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            vis = (vis * 0.3).astype(np.uint8)  # Darken for better contrast
        else:
            vis = np.zeros((original.shape[0], original.shape[1], 3), dtype=np.uint8)
            vis[:, :] = [30, 30, 30]  # Dark gray background

        # Calculate differences
        added = (refined > 0) & (original == 0)  # New pixels in refined
        removed = (original > 0) & (refined == 0)  # Pixels removed in refined
        unchanged = (original > 0) & (refined > 0)  # Common pixels

        # Apply colors
        vis[unchanged] = [200, 200, 200]  # Light gray for unchanged mask
        vis[added] = [0, 255, 0]  # Green for additions
        vis[removed] = [0, 0, 255]  # Red for removals

        # Add contours for clarity
        orig_contours, _ = cv2.findContours(original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(vis, orig_contours, -1, (255, 100, 100), 1)  # Light red for original
        cv2.drawContours(vis, ref_contours, -1, (100, 255, 100), 1)  # Light green for refined

        # Add statistics text
        num_added = np.sum(added)
        num_removed = np.sum(removed)
        area_change = (np.sum(refined > 0) - np.sum(original > 0)) / (np.sum(original > 0) + 1e-6)

        # Add text with statistics
        cv2.putText(vis, f"Added: {num_added} px", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Removed: {num_removed} px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Area change: {area_change*100:.1f}%", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def batch_refine(self, mask_dir: str, output_dir: str,
                    image_dir: Optional[str] = None,
                    save_visualizations: bool = True,
                    method: str = 'hybrid') -> Dict:
        """
        Batch process masks with refinement.

        Args:
            mask_dir: Directory containing masks
            output_dir: Directory for refined masks
            image_dir: Optional directory with ultrasound images
            save_visualizations: Whether to save difference visualizations
            method: Refinement method to use

        Returns:
            summary: Dictionary with processing statistics
        """
        mask_dir = Path(mask_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_visualizations:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)

        # Find all mask files
        mask_files = list(mask_dir.glob('*_mask.png')) + list(mask_dir.glob('*.png'))

        summary = {
            'total': len(mask_files),
            'improved': 0,
            'unchanged': 0,
            'avg_area_change': 0,
            'avg_hausdorff': 0
        }

        for mask_path in mask_files:
            # Load mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Refine
            refined, metrics = self.refine(mask, method=method)

            # Update summary
            if metrics['area_change'] > 0.001 or metrics['hausdorff'] > 0.5:
                summary['improved'] += 1
            else:
                summary['unchanged'] += 1

            summary['avg_area_change'] += metrics['area_change']
            summary['avg_hausdorff'] += metrics['hausdorff']

            # Save refined mask
            output_path = output_dir / mask_path.name
            cv2.imwrite(str(output_path), refined)

            # Create visualization if requested
            if save_visualizations:
                # Try to load corresponding image
                image = None
                if image_dir:
                    image_name = mask_path.name.replace('_mask', '').replace('.png', '') + '.png'
                    image_path = Path(image_dir) / image_name
                    if image_path.exists():
                        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

                vis = self.create_difference_visualization(mask, refined, image)
                vis_path = vis_dir / f"{mask_path.stem}_diff.png"
                cv2.imwrite(str(vis_path), vis)

            print(f"Processed {mask_path.name}: "
                  f"area_change={metrics['area_change']:.3f}, "
                  f"hausdorff={metrics['hausdorff']:.1f}")

        # Calculate averages
        if summary['total'] > 0:
            summary['avg_area_change'] /= summary['total']
            summary['avg_hausdorff'] /= summary['total']

        return summary


def main():
    """Simple command line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Refine ultrasound segmentation masks')
    parser.add_argument('--masks', default='Bladder_Data/masks', help='Mask directory')
    parser.add_argument('--output', default='Bladder_Data/refined_masks', help='Output directory')
    parser.add_argument('--images', default='Bladder_Data/images', help='Image directory for visualizations')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualizations')

    args = parser.parse_args()

    # Use moderate settings to actually smooth jagged contours
    refiner = UnifiedMaskRefiner(
        min_area_change=0.05,
        max_hausdorff_increase=8.0,
        smoothing_strength='moderate'
    )

    print("Refining masks...")
    summary = refiner.batch_refine(
        mask_dir=args.masks,
        output_dir=args.output,
        image_dir=None if args.no_vis else args.images,
        save_visualizations=not args.no_vis,
        method='hybrid'
    )

    print(f"\nDone! {summary['improved']}/{summary['total']} masks improved")
    print(f"Average change: {summary['avg_area_change']*100:.1f}%")


if __name__ == '__main__':
    main()