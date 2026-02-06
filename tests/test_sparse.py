"""
Tests for sparse and non-sparse data format support
"""
import numpy as np
import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import brimfile as brim

from datetime import datetime

def lorentzian(x, x0, w):
    return 1/(1+((x-x0)/(w/2))**2)

def generate_test_data():
    """Generate test data for both sparse and non-sparse formats"""
    Nx, Ny, Nz = (7, 5, 3)  # Number of points in x,y,z
    dx, dy, dz = (0.4, 0.5, 2)  # Stepsizes (in um)
    n_points = Nx*Ny*Nz  # total number of points

    width_GHz = 0.4
    width_GHz_arr = np.full((Nz, Ny, Nx), width_GHz)
    shift_GHz_arr = np.empty((Nz, Ny, Nx))
    freq_GHz = np.linspace(6, 9, 151)  # 151 frequency points
    PSD = np.empty((Nz, Ny, Nx, len(freq_GHz)))
    
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                index = k + Nx*j + Ny*Nx*i
                shift_GHz = freq_GHz[0] + (freq_GHz[-1]-freq_GHz[0]) * index/n_points
                spectrum = lorentzian(freq_GHz, shift_GHz, width_GHz)
                shift_GHz_arr[i,j,k] = shift_GHz 
                PSD[i, j, k,:] = spectrum

    return PSD, freq_GHz, (dz,dy,dx), shift_GHz_arr, width_GHz_arr

def test_non_sparse_creation_and_reading():
    """Test creating and reading non-sparse data (new format)"""
    print("Testing non-sparse data format...")
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'file_nonsparse.brim.zarr'))
    
    # Clean up if file exists
    if os.path.exists(filename):
        shutil.rmtree(filename)
    
    try:
        # Create file with non-sparse data
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_test_data()
        
        # Create data group with sparse=False (new format)
        d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='nonsparse_test', sparse=False)
        
        # Check that the data is recognized as non-sparse
        assert not d0.is_sparse(), "Data should be non-sparse"
        
        # Add metadata
        Attr = brim.Metadata.Item
        datetime_now = datetime.now().isoformat()
        md = d0.get_metadata()
        md.add(brim.Metadata.Type.Experiment, {'Datetime':datetime_now})
        
        # Create analysis results
        ar = d0.create_analysis_results_group(
            {'shift':shift_GHz, 'shift_units': 'GHz', 'width': width_GHz, 'width_units': 'Hz'},
            None,
            name='test_analysis',
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
        )
        
        f.close()
        
        # Read back the file
        f = brim.File(filename)
        
        # Get the data group
        d = f.get_data()
        
        # Verify it's non-sparse
        assert not d.is_sparse(), "Data should still be non-sparse after reading"
        
        # Get PSD in spatial map format
        PSD_read, freq_read, _, _ = d.get_PSD_as_spatial_map()
        
        # Verify shape is 4D
        assert PSD_read.ndim == 4, f"Expected 4D PSD, got {PSD_read.ndim}D"
        assert PSD_read.shape == PSD.shape, f"Shape mismatch: {PSD_read.shape} vs {PSD.shape}"
        
        # Verify data matches
        assert np.allclose(PSD_read, PSD), "PSD data mismatch"
        assert np.allclose(freq_read, freq_GHz), "Frequency data mismatch"
        
        # Get spectrum at a specific coordinate
        coord = (0, 0, 0)
        PSD_coord, freq_coord, _, _ = d.get_spectrum_in_image(coord)
        assert np.allclose(PSD_coord, PSD[coord]), "Spectrum at coordinate mismatch"
        
        # Get analysis results
        ar = d.get_analysis_results()
        img, px_size = ar.get_image(brim.Data.AnalysisResults.Quantity.Shift, 
                                     brim.Data.AnalysisResults.PeakType.AntiStokes)
        
        # Verify image shape
        assert img.shape == shift_GHz.shape, f"Image shape mismatch: {img.shape} vs {shift_GHz.shape}"
        assert np.allclose(img, shift_GHz), "Image data mismatch"
        
        f.close()
        
        print("✓ Non-sparse data format test passed!")
        return True
        
    finally:
        # Clean up
        if os.path.exists(filename):
            shutil.rmtree(filename)

def test_sparse_backward_compatibility():
    """Test that sparse data (old format) still works"""
    print("Testing sparse data (backward compatibility)...")
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'file_sparse.brim.zarr'))
    
    # Clean up if file exists
    if os.path.exists(filename):
        shutil.rmtree(filename)
    
    try:
        # Create file with sparse data (default behavior)
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_test_data()
        
        # Create data group with default (sparse=None means sparse=True for backward compatibility)
        d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='sparse_test')
        
        # Check that the data is recognized as sparse
        assert d0.is_sparse(), "Data should be sparse"
        
        # Create analysis results
        ar = d0.create_analysis_results_group(
            {'shift':shift_GHz, 'shift_units': 'GHz', 'width': width_GHz, 'width_units': 'Hz'},
            None,
            name='test_analysis',
            fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
        )
        
        f.close()
        
        # Read back the file
        f = brim.File(filename)
        d = f.get_data()
        
        # Verify it's sparse
        assert d.is_sparse(), "Data should still be sparse after reading"
        
        # Get PSD in spatial map format (should be reshaped to 4D)
        PSD_read, freq_read, _, _ = d.get_PSD_as_spatial_map()
        
        # Verify shape is 4D after reconstruction
        assert PSD_read.ndim == 4, f"Expected 4D PSD after reconstruction, got {PSD_read.ndim}D"
        assert PSD_read.shape == PSD.shape, f"Shape mismatch: {PSD_read.shape} vs {PSD.shape}"
        
        # Verify data matches
        assert np.allclose(PSD_read, PSD), "PSD data mismatch"
        
        # Get analysis results
        ar = d.get_analysis_results()
        img, px_size = ar.get_image(brim.Data.AnalysisResults.Quantity.Shift, 
                                     brim.Data.AnalysisResults.PeakType.AntiStokes)
        
        # Verify image shape
        assert img.shape == shift_GHz.shape, f"Image shape mismatch: {img.shape} vs {shift_GHz.shape}"
        assert np.allclose(img, shift_GHz), "Image data mismatch"
        
        f.close()
        
        print("✓ Sparse data (backward compatibility) test passed!")
        return True
        
    finally:
        # Clean up
        if os.path.exists(filename):
            shutil.rmtree(filename)

def test_explicit_sparse_true():
    """Test creating data with explicit sparse=True"""
    print("Testing explicit sparse=True...")
    
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), 'file_explicit_sparse.brim.zarr'))
    
    # Clean up if file exists
    if os.path.exists(filename):
        shutil.rmtree(filename)
    
    try:
        f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
        
        PSD, freq_GHz, (dz,dy,dx), shift_GHz, width_GHz = generate_test_data()
        
        # Create data group with explicit sparse=True
        d0 = f.create_data_group(PSD, freq_GHz, (dz,dy,dx), name='explicit_sparse', sparse=True)
        
        # Check that the data is recognized as sparse
        assert d0.is_sparse(), "Data should be sparse"
        
        f.close()
        
        # Read back and verify
        f = brim.File(filename)
        d = f.get_data()
        assert d.is_sparse(), "Data should be sparse after reading"
        
        f.close()
        
        print("✓ Explicit sparse=True test passed!")
        return True
        
    finally:
        if os.path.exists(filename):
            shutil.rmtree(filename)

if __name__ == "__main__":
    all_passed = True
    
    try:
        all_passed &= test_non_sparse_creation_and_reading()
    except Exception as e:
        print(f"✗ Non-sparse test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_sparse_backward_compatibility()
    except Exception as e:
        print(f"✗ Sparse backward compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_explicit_sparse_true()
    except Exception as e:
        print(f"✗ Explicit sparse=True test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
