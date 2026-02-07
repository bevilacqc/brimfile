"""
Unit tests for the Data class in brimfile.
"""


import numpy as np

import brimfile as brim


class TestDataProperties:
    """Tests for Data class properties and basic methods."""
    
    def test_get_name(self, simple_brim_file):
        """Test getting the name of a data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        name = data.get_name()
        assert name is not None
        assert isinstance(name, str)
        f.close()
    
    def test_get_index(self, simple_brim_file):
        """Test getting the index of a data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        index = data.get_index()
        assert isinstance(index, int)
        assert index >= 0
        f.close()
    
    def test_get_num_parameters(self, simple_brim_file):
        """Test getting the number of parameters."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        n_pars = data.get_num_parameters()
        assert isinstance(n_pars, tuple)
        f.close()


class TestSpectrumRetrieval:
    """Tests for spectrum retrieval from Data."""
    
    def test_get_spectrum_in_image(self, simple_brim_file):
        """Test getting spectrum at a specific pixel."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coord = (0, 0, 0)
        PSD, frequency, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        assert PSD is not None
        assert frequency is not None
        assert isinstance(PSD, np.ndarray)
        assert isinstance(frequency, np.ndarray)
        assert len(PSD) == len(frequency)
        f.close()
    
    def test_get_spectrum_different_coordinates(self, simple_brim_file):
        """Test getting spectra at different coordinates."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coords = [(0, 0, 0), (1, 2, 3), (2, 4, 6)]
        for coord in coords:
            PSD, frequency, _, _ = data.get_spectrum_in_image(coord)
            assert PSD is not None
            assert len(PSD) > 0
        
        f.close()
    
    def test_spectrum_units(self, simple_brim_file):
        """Test that spectrum units are returned."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        
        coord = (0, 0, 0)
        _, _, PSD_units, frequency_units = data.get_spectrum_in_image(coord)
        
        # Units should be strings or None
        assert PSD_units is None or isinstance(PSD_units, str)
        assert frequency_units is None or isinstance(frequency_units, str)
        f.close()


class TestMetadataAccess:
    """Tests for metadata access from Data."""
    
    def test_get_metadata(self, simple_brim_file):
        """Test getting metadata from data group."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        metadata = data.get_metadata()
        
        assert metadata is not None
        assert isinstance(metadata, brim.Metadata)
        f.close()


class TestAnalysisResults:
    """Tests for analysis results in Data."""
    
    def test_list_analysis_results(self, simple_brim_file):
        """Test listing analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar_list = data.list_AnalysisResults()
        
        assert ar_list is not None
        assert len(ar_list) > 0
        f.close()
    
    def test_list_analysis_results_with_names(self, simple_brim_file):
        """Test listing analysis results with custom names."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar_results = data.list_AnalysisResults(retrieve_custom_name=True)
        
        assert ar_results is not None
        assert isinstance(ar_results, list)
        f.close()
    
    def test_get_analysis_results(self, simple_brim_file):
        """Test getting analysis results."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results()
        
        assert ar is not None
        assert isinstance(ar, brim.Data.AnalysisResults)
        f.close()
    
    def test_get_analysis_results_by_index(self, simple_brim_file):
        """Test getting analysis results by index."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        ar = data.get_analysis_results(0)
        
        assert ar is not None
        f.close()
    
    def test_create_analysis_results(self, empty_brim_file, sample_data):
        """Test creating new analysis results."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size']
        )
        
        ar = data.create_analysis_results_group(
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
            name='test_ar'
        )
        
        assert ar is not None
        f.close()


class TestDataCreation:
    """Tests for creating data groups with different parameters."""
    
    def test_create_with_custom_name(self, empty_brim_file, sample_data):
        """Test creating data group with custom name."""
        f = brim.File(empty_brim_file, mode='r+')
        data = f.create_data_group(
            sample_data['PSD'],
            sample_data['frequency'],
            sample_data['pixel_size'],
            name='custom_name'
        )
        
        name = data.get_name()
        assert 'custom_name' in name or name == 'custom_name'
        f.close()
    
    def test_create_with_different_dimensions(self, empty_brim_file):
        """Test creating data with different dimensions."""
        f = brim.File(empty_brim_file, mode='r+')
        
        # Smaller dataset
        PSD = np.random.rand(2, 3, 4, 50)
        frequency = np.linspace(5, 10, 50)
        pixel_size = (1.0, 1.0, 1.0)
        
        data = f.create_data_group(PSD, frequency, pixel_size)
        assert data is not None
        f.close()
