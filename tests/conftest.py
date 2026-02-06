"""
Pytest configuration and shared fixtures for brimfile tests.
"""

import pytest
import numpy as np
import os
import shutil
import tempfile
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import brimfile as brim


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


@pytest.fixture
def sample_data():
    """Generate sample spectral data for testing."""
    def lorentzian(x, x0, w):
        return 1 / (1 + ((x - x0) / (w / 2)) ** 2)
    
    Nx, Ny, Nz = (7, 5, 3)  # Number of points in x, y, z
    dx, dy, dz = (0.4, 0.5, 2)  # Step sizes (in µm)
    n_points = Nx * Ny * Nz
    
    width_GHz = 0.4
    width_GHz_arr = np.full((Nz, Ny, Nx), width_GHz)
    shift_GHz_arr = np.empty((Nz, Ny, Nx))
    freq_GHz = np.linspace(6, 9, 151)  # 151 frequency points
    PSD = np.empty((Nz, Ny, Nx, len(freq_GHz)))
    
    for i in range(Nz):
        for j in range(Ny):
            for k in range(Nx):
                index = k + Nx * j + Ny * Nx * i
                # Increase shift linearly
                shift_GHz = freq_GHz[0] + (freq_GHz[-1] - freq_GHz[0]) * index / n_points
                spectrum = lorentzian(freq_GHz, shift_GHz, width_GHz)
                shift_GHz_arr[i, j, k] = shift_GHz
                PSD[i, j, k, :] = spectrum
    
    return {
        'PSD': PSD,
        'frequency': freq_GHz,
        'pixel_size': (dz, dy, dx),
        'shift': shift_GHz_arr,
        'width': width_GHz_arr,
        'dimensions': (Nz, Ny, Nx)
    }


@pytest.fixture
def simple_brim_file(temp_dir, sample_data):
    """Create a simple brim file for testing."""
    filename = os.path.join(temp_dir, 'test_file.brim.zarr')
    
    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
    
    # Create data group
    d = f.create_data_group(
        sample_data['PSD'],
        sample_data['frequency'],
        sample_data['pixel_size'],
        name='test_data'
    )
    
    # Add basic metadata
    Attr = brim.Metadata.Item
    datetime_now = datetime.now().isoformat()
    temp = Attr(22.0, 'C')
    md = d.get_metadata()
    md.add(brim.Metadata.Type.Experiment, {'Datetime': datetime_now, 'Temperature': temp})
    md.add(brim.Metadata.Type.Optics, {'Wavelength': Attr(660, 'nm')})
    
    # Create analysis results
    ar = d.create_analysis_results_group(
        {
            'shift': sample_data['shift'],
            'shift_units': 'GHz',
            'width': sample_data['width'],
            'width_units': 'GHz'
        },
        {
            'shift': sample_data['shift'],
            'shift_units': 'GHz',
            'width': sample_data['width'],
            'width_units': 'GHz'
        },
        name='test_analysis',
        fit_model=brim.Data.AnalysisResults.FitModel.Lorentzian
    )
    
    f.close()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        shutil.rmtree(filename)


@pytest.fixture
def empty_brim_file(temp_dir):
    """Create an empty brim file for testing."""
    filename = os.path.join(temp_dir, 'empty_file.brim.zarr')
    f = brim.File.create(filename, store_type=brim.StoreType.AUTO)
    f.close()
    
    yield filename
    
    # Cleanup
    if os.path.exists(filename):
        shutil.rmtree(filename)
