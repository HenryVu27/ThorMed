# Post-Processing Mask Refinement Plan for Bladder Ultrasound Segmentation

## Overview
This plan provides a comprehensive post-processing pipeline that combines the strengths of both existing implementations while focusing exclusively on mask refinement (no preprocessing). The goal is to produce training-grade segmentation masks with smooth boundaries while preserving anatomical accuracy.

---

## Critical Analysis of Existing Implementations

### Current Implementation (`mask_refiner.py`)
**Strengths:**
- Fast, deterministic, simple to batch
- Optional spline smoothing
- Handy visualization tools
- Decent default parameters
- Comprehensive error handling

**Critical Issues:**
- Mask-only approach (no edge snapping to image features)
- Gaussian + hard threshold can shift boundaries
- Spline smoothing can bow bladder neck and bridge concavities
- Largest-component pruning can drop legitimate second lobes
- No guardrails (area drift, HD95) to prevent degradations

### Independent Implementation (`refinement_plan.md`)
**Strengths:**
- Mandatory audit step for data quality
- Three-stage cascade (morphology → CRF → Fourier descriptors)
- Dense-CRF adds image-guided edge snapping
- Fourier descriptor smoothing reduces high-frequency jaggies
- Includes skip logic and verification metrics

**Critical Issues:**
- CRF/FD parameters are sensitive
- CRF may over-snap to speckle edges on some scans
- FD can over-round neck/invaginations if keep is too low
- Needs stronger safety gates (area/HD95/fallback)

---

## Recommended Post-Processing Cascade

### Stage 0: Mandatory Audit (Keep from Independent Plan)
**Purpose:** Ensure data quality and prevent garbage-in effects downstream.

```python
def audit_mask(mask_path):
    """
    Audit SAM-generated bladder masks for pixel integrity and topology.
    Returns: binary_enforced_mask, audit_metrics
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Enforce binary through threshold
    binary_mask = (mask > 127).astype(np.uint8) * 255
    
    # Calculate metrics
    area = np.sum(binary_mask > 0)
    num_components = cv2.connectedComponents(binary_mask)[0] - 1
    holes = cv2.countNonZero(255 - cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8)))
    
    audit_metrics = {
        'area_px': area,
        'components': num_components,
        'holes_px': holes,
        'needs_attention': num_components > 1 or holes > 0
    }
    
    return binary_mask, audit_metrics
```

### Stage A: Conservative Morphology with Component Logic
**Purpose:** Remove noise and fill small holes while preserving anatomical structures.

```python
def adaptive_morphology(mask, ultrasound_image=None):
    """
    Apply adaptive morphological operations with component-aware logic.
    """
    # Calculate adaptive parameters
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    
    # Adaptive kernel size
    kernel_size = max(3, min(7, round(perimeter / 150)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Adaptive hole filling threshold
    hole_threshold = max(30, int(0.01 * area))
    
    # Apply operations
    # 1. Fill small holes
    filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # 2. Opening then closing
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 3. Component-aware filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    
    if num_labels <= 1:
        return closed
    
    # Keep largest component
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    
    # Check for legitimate second components
    for label in range(2, num_labels):
        component_area = stats[label, cv2.CC_STAT_AREA]
        if component_area >= 0.02 * largest_area:  # 2% threshold
            # Check if it's close to main component
            component_mask = (labels == label).astype(np.uint8) * 255
            dilated_main = cv2.dilate((labels == largest_label).astype(np.uint8) * 255, 
                                    np.ones((5, 5), np.uint8))
            if cv2.countNonZero(cv2.bitwise_and(component_mask, dilated_main)) > 0:
                # Merge with main component
                largest_area += component_area
            else:
                # Keep as separate component
                pass
    
    # Create final mask
    result = np.zeros_like(closed)
    result[labels == largest_label] = 255
    
    return result
```

### Stage B: Image-Guided CRF Edge Snap
**Purpose:** Snap boundaries to actual image features while preserving smoothness.

```python
def crf_edge_snap(mask, ultrasound_image, iterations=5):
    """
    Apply Dense-CRF for image-guided edge snapping.
    """
    import pydensecrf.densecrf as dcrf
    import pydensecrf.utils as utils
    
    h, w = mask.shape[:2]
    
    # Initialize CRF
    d = dcrf.DenseCRF2D(w, h, 2)
    
    # Set unary potentials
    unary = utils.unary_from_labels(mask // 255, 2, gt_prob=0.8)
    d.setUnaryEnergy(unary)
    
    # Add pairwise potentials
    # Gaussian (appearance-independent)
    d.addPairwiseGaussian(sxy=3, compat=3)
    
    # Bilateral (appearance-dependent)
    d.addPairwiseBilateral(sxy=20, srgb=10, rgbim=ultrasound_image, compat=10)
    
    # Inference
    Q = d.inference(iterations)
    
    # Convert back to binary mask
    refined_mask = (np.argmax(Q, 0).reshape(h, w) * 255).astype(np.uint8)
    
    return refined_mask

def conditional_crf(mask, ultrasound_image, jaccard_threshold=0.02):
    """
    Apply CRF only if significant changes are expected.
    """
    # Calculate expected change using simple heuristics
    # (This is a simplified version - could be more sophisticated)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    
    # Estimate boundary roughness
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    area = cv2.contourArea(largest_contour)
    roughness = perimeter**2 / (4 * np.pi * area)  # Circularity measure
    
    # Apply CRF if boundary is rough
    if roughness > 1.2:  # Threshold for "rough" boundary
        return crf_edge_snap(mask, ultrasound_image)
    else:
        return mask
```

### Stage C: Contour Smoothing with Safety Nets
**Purpose:** Reduce high-frequency jaggies while preserving anatomical features.

```python
def fourier_descriptor_smoothing(mask, keep_ratio=1/70):
    """
    Apply Fourier descriptor smoothing with adaptive parameters.
    """
    from skimage import measure
    
    # Find contours
    contours = measure.find_contours(mask, 0.5)
    if not contours:
        return mask
    
    # Use largest contour
    largest_contour = max(contours, key=len)
    
    # Adaptive keep parameter
    contour_length = len(largest_contour)
    keep = max(20, min(60, int(contour_length * keep_ratio)))
    
    # Apply Fourier smoothing
    fx = np.fft.fft(largest_contour[:, 1])
    fy = np.fft.fft(largest_contour[:, 0])
    
    # Zero out high-frequency components
    fx[keep:-keep] = 0
    fy[keep:-keep] = 0
    
    # Inverse FFT
    smoothed_y = np.fft.ifft(fx).real
    smoothed_x = np.fft.ifft(fy).real
    
    # Create smoothed contour
    smoothed_contour = np.stack([smoothed_x, smoothed_y], axis=1).round().astype(int)
    
    # Create new mask
    smoothed_mask = np.zeros_like(mask)
    cv2.fillPoly(smoothed_mask, [smoothed_contour], 255)
    
    return smoothed_mask

def spline_smoothing_safe(mask, smoothing_factor=0.1, boundary_band=5):
    """
    Apply spline smoothing only within a boundary band to prevent global warping.
    """
    from scipy.interpolate import splprep, splev
    
    # Create trimap (sure foreground, unknown boundary, sure background)
    kernel = np.ones((boundary_band, boundary_band), np.uint8)
    eroded = cv2.erode(mask, kernel)
    dilated = cv2.dilate(mask, kernel)
    
    # Unknown region is the boundary band
    unknown = cv2.subtract(dilated, eroded)
    
    # Apply spline smoothing only in unknown region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    if len(largest_contour) < 4:
        return mask
    
    # Spline fitting
    x, y = largest_contour.squeeze().T
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]
    
    tck, u = splprep([x, y], s=smoothing_factor * len(x), per=True)
    
    # Evaluate at higher resolution
    unew = np.linspace(0, 1, 1000)
    out = splev(unew, tck)
    
    # Create smoothed contour
    smoothed_contour = np.array([out]).T.reshape((-1, 1, 2)).astype(np.int32)
    
    # Create new mask
    smoothed_mask = np.zeros_like(mask)
    cv2.drawContours(smoothed_mask, [smoothed_contour], -1, 255, thickness=cv2.FILLED)
    
    # Blend with original in sure foreground/background regions
    result = mask.copy()
    result[unknown > 0] = smoothed_mask[unknown > 0]
    
    return result
```

### Stage D: Final Cleanup
**Purpose:** Remove any artifacts introduced by previous stages.

```python
def final_cleanup(mask):
    """
    Final cleanup to ensure binary mask quality.
    """
    # Remove single-pixel spikes
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel)
    cleaned = cv2.dilate(eroded, kernel)
    
    # Fill any remaining small holes
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Ensure binary
    cleaned = (cleaned > 127).astype(np.uint8) * 255
    
    return cleaned
```

### Stage E: Verification and Fallback
**Purpose:** Ensure quality metrics are met, with fallback to previous stages if needed.

```python
def calculate_metrics(original_mask, refined_mask):
    """
    Calculate quality metrics for refinement validation.
    """
    # Area drift
    original_area = np.sum(original_mask > 0)
    refined_area = np.sum(refined_mask > 0)
    area_drift = abs(refined_area - original_area) / original_area
    
    # Perimeter change
    original_contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refined_contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    original_perimeter = cv2.arcLength(original_contours[0], True) if original_contours else 0
    refined_perimeter = cv2.arcLength(refined_contours[0], True) if refined_contours else 0
    perimeter_change = (original_perimeter - refined_perimeter) / original_perimeter if original_perimeter > 0 else 0
    
    # Hausdorff distance (simplified)
    # Note: Full HD95 calculation would require more sophisticated implementation
    hd95_estimate = estimate_hausdorff_distance(original_mask, refined_mask)
    
    return {
        'area_drift': area_drift,
        'perimeter_change': perimeter_change,
        'hd95_estimate': hd95_estimate
    }

def validate_refinement(original_mask, refined_mask, stage_name):
    """
    Validate refinement results and potentially fallback.
    """
    metrics = calculate_metrics(original_mask, refined_mask)
    
    # Quality thresholds
    area_drift_threshold = 0.03  # 3%
    perimeter_change_threshold = 0.25  # 25%
    
    # Check if refinement is acceptable
    is_valid = (
        metrics['area_drift'] <= area_drift_threshold and
        metrics['perimeter_change'] <= perimeter_change_threshold
    )
    
    if not is_valid:
        print(f"Warning: {stage_name} failed quality checks. Metrics: {metrics}")
        return False, metrics
    
    return True, metrics
```

---

## Complete Pipeline Implementation

```python
def refine_mask_complete(mask_path, ultrasound_path, output_path):
    """
    Complete post-processing pipeline with all stages and safety checks.
    """
    # Load images
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ultrasound = cv2.imread(ultrasound_path)
    
    # Stage 0: Audit
    original_mask, audit_metrics = audit_mask(mask_path)
    if audit_metrics['needs_attention']:
        print(f"Warning: Mask {mask_path} needs attention: {audit_metrics}")
    
    current_mask = original_mask.copy()
    stage_results = {}
    
    # Stage A: Morphology
    try:
        stage_a_mask = adaptive_morphology(current_mask)
        is_valid, metrics = validate_refinement(current_mask, stage_a_mask, "Stage A")
        if is_valid:
            current_mask = stage_a_mask
            stage_results['A'] = metrics
        else:
            print("Stage A failed validation, keeping original")
    except Exception as e:
        print(f"Stage A failed: {e}")
    
    # Stage B: CRF (if ultrasound available)
    if ultrasound is not None:
        try:
            stage_b_mask = conditional_crf(current_mask, ultrasound)
            is_valid, metrics = validate_refinement(current_mask, stage_b_mask, "Stage B")
            if is_valid:
                current_mask = stage_b_mask
                stage_results['B'] = metrics
            else:
                print("Stage B failed validation, keeping previous stage")
        except Exception as e:
            print(f"Stage B failed: {e}")
    
    # Stage C: Contour Smoothing
    try:
        # Try Fourier descriptors first
        stage_c_mask = fourier_descriptor_smoothing(current_mask)
        is_valid, metrics = validate_refinement(current_mask, stage_c_mask, "Stage C (FD)")
        
        if not is_valid:
            # Fallback to safe spline smoothing
            stage_c_mask = spline_smoothing_safe(current_mask)
            is_valid, metrics = validate_refinement(current_mask, stage_c_mask, "Stage C (Spline)")
        
        if is_valid:
            current_mask = stage_c_mask
            stage_results['C'] = metrics
        else:
            print("Stage C failed validation, keeping previous stage")
    except Exception as e:
        print(f"Stage C failed: {e}")
    
    # Stage D: Final Cleanup
    try:
        final_mask = final_cleanup(current_mask)
        is_valid, metrics = validate_refinement(current_mask, final_mask, "Stage D")
        if is_valid:
            current_mask = final_mask
            stage_results['D'] = metrics
        else:
            print("Stage D failed validation, keeping previous stage")
    except Exception as e:
        print(f"Stage D failed: {e}")
    
    # Save result
    cv2.imwrite(output_path, current_mask)
    
    # Return summary
    return {
        'output_path': output_path,
        'stages_applied': list(stage_results.keys()),
        'final_metrics': stage_results,
        'audit_metrics': audit_metrics
    }
```

---

## Parameter Guidelines

### Adaptive Parameters
- **Kernel Size**: `k = clamp(perimeter/150, 3, 7)`
- **Hole Fill Threshold**: `max(30 px, 0.01 × foreground_area)`
- **Fourier Keep**: `clamp(contour_length/70, 20, 60)`
- **Spline Smoothing**: `smoothing_factor = 0.05-0.15`

### Quality Thresholds
- **Area Drift**: ≤ 3%
- **Perimeter Change**: ≤ 25%
- **Component Merge**: ≥ 2% of largest component area
- **CRF Skip**: Jaccard change < 2%

### CRF Parameters (Ultrasound-optimized)
- **Iterations**: 5-10
- **Gaussian**: sxy=3, compat=3
- **Bilateral**: sxy=20, srgb=10, compat=10

---

## Integration with Existing Code

To integrate this plan with your existing `mask_refiner.py`:

1. **Add the audit function** as a mandatory first step
2. **Replace the current morphological operations** with the adaptive version
3. **Add optional CRF step** that takes ultrasound images as input
4. **Replace Gaussian smoothing** with Fourier descriptors or safe spline smoothing
5. **Add validation metrics** after each stage
6. **Implement fallback logic** to previous stages on validation failure

This approach maintains the speed and simplicity of your current implementation while adding the image-guided refinement and safety checks from the independent plan.

