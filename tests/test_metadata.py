"""
Unit tests for the Metadata class in brimfile.
"""

import pytest
from datetime import datetime

import brimfile as brim


class TestMetadataItem:
    """Tests for Metadata.Item class."""
    
    def test_create_item_with_units(self):
        """Test creating a metadata item with units."""
        item = brim.Metadata.Item(22.0, 'C')
        assert item.value == 22.0
        assert item.units == 'C'
    
    def test_create_item_without_units(self):
        """Test creating a metadata item without units."""
        item = brim.Metadata.Item(100)
        assert item.value == 100
        assert item.units is None
    
    def test_item_string_representation(self):
        """Test string representation of metadata item."""
        item = brim.Metadata.Item(660, 'nm')
        str_rep = str(item)
        assert '660' in str_rep
        assert 'nm' in str_rep


class TestMetadataAddition:
    """Tests for adding metadata."""
    
    def test_add_experiment_metadata(self, simple_brim_file):
        """Test adding experiment metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        temp = Attr(25.0, 'C')
        md.add(brim.Metadata.Type.Experiment, {'Temperature': temp}, local=True)
        
        # Verify it was added
        retrieved_temp = md['Experiment.Temperature']
        assert retrieved_temp.value == 25.0
        assert retrieved_temp.units == 'C'
        f.close()
    
    def test_add_optics_metadata(self, simple_brim_file):
        """Test adding optics metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        wavelength = Attr(532, 'nm')
        md.add(brim.Metadata.Type.Optics, {'Wavelength': wavelength}, local=True)
        
        retrieved = md['Optics.Wavelength']
        assert retrieved.value == 532
        f.close()
    
    def test_add_global_metadata(self, simple_brim_file):
        """Test adding global (non-local) metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        md.add(
            brim.Metadata.Type.Experiment,
            {'Operator': 'Test User'},
            local=False
        )
        f.close()
    
    def test_add_datetime_metadata(self, simple_brim_file):
        """Test adding datetime metadata."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        datetime_now = datetime.now().isoformat()
        md.add(
            brim.Metadata.Type.Experiment,
            {'Datetime': datetime_now},
            local=True
        )
        
        retrieved = md['Experiment.Datetime']
        assert retrieved.value == datetime_now
        f.close()


class TestMetadataRetrieval:
    """Tests for retrieving metadata."""
    
    def test_retrieve_single_metadata(self, simple_brim_file):
        """Test retrieving a single metadata item."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        # Should have temperature from fixture
        temp = md['Experiment.Temperature']
        assert temp is not None
        assert temp.value == 22.0
        assert temp.units == 'C'
        f.close()
    
    def test_retrieve_wavelength(self, simple_brim_file):
        """Test retrieving wavelength metadata."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        wavelength = md['Optics.Wavelength']
        assert wavelength is not None
        assert wavelength.value == 660
        assert wavelength.units == 'nm'
        f.close()
    
    def test_retrieve_nonexistent_metadata(self, simple_brim_file):
        """Test that retrieving non-existent metadata raises KeyError."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        with pytest.raises(KeyError):
            _ = md['Experiment.NonExistent']
        f.close()


class TestMetadataDictConversion:
    """Tests for converting metadata to dictionaries."""
    
    def test_all_to_dict(self, simple_brim_file):
        """Test converting all metadata to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        all_md = md.all_to_dict()
        assert all_md is not None
        assert isinstance(all_md, dict)
        f.close()
    
    def test_to_dict_by_type(self, simple_brim_file):
        """Test converting metadata of specific type to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        exp_md = md.to_dict(brim.Metadata.Type.Experiment)
        assert exp_md is not None
        assert isinstance(exp_md, dict)
        f.close()
    
    def test_to_dict_optics(self, simple_brim_file):
        """Test converting optics metadata to dictionary."""
        f = brim.File(simple_brim_file)
        data = f.get_data()
        md = data.get_metadata()
        
        optics_md = md.to_dict(brim.Metadata.Type.Optics)
        assert optics_md is not None
        assert 'Wavelength' in optics_md
        f.close()


class TestMetadataTypes:
    """Tests for different metadata types."""
    
    def test_experiment_type(self):
        """Test Experiment metadata type."""
        assert brim.Metadata.Type.Experiment.value == 'Experiment'
    
    def test_optics_type(self):
        """Test Optics metadata type."""
        assert brim.Metadata.Type.Optics.value == 'Optics'
    
    def test_brillouin_type(self):
        """Test Brillouin metadata type."""
        assert brim.Metadata.Type.Brillouin.value == 'Brillouin'
    
    def test_acquisition_type(self):
        """Test Acquisition metadata type."""
        assert brim.Metadata.Type.Acquisition.value == 'Acquisition'
    
    def test_spectrometer_type(self):
        """Test Spectrometer metadata type."""
        assert brim.Metadata.Type.Spectrometer.value == 'Spectrometer'


class TestMetadataUpdate:
    """Tests for updating existing metadata."""
    
    def test_update_metadata_value(self, simple_brim_file):
        """Test updating an existing metadata value."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        # Add initial value
        Attr = brim.Metadata.Item
        md.add(brim.Metadata.Type.Experiment, {'Temperature': Attr(20.0, 'C')}, local=True)
        
        # Update with new value
        md.add(brim.Metadata.Type.Experiment, {'Temperature': Attr(25.0, 'C')}, local=True)
        
        # Check updated value
        temp = md['Experiment.Temperature']
        assert temp.value == 25.0
        f.close()


class TestLocalVsGlobalMetadata:
    """Tests for local vs global metadata handling."""
    
    def test_local_metadata_in_data_group(self, simple_brim_file):
        """Test that local metadata is specific to data group."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        Attr = brim.Metadata.Item
        md.add(
            brim.Metadata.Type.Experiment,
            {'LocalValue': Attr(100, 'units')},
            local=True
        )
        
        # Should be retrievable from this data group
        local_val = md['Experiment.LocalValue']
        assert local_val.value == 100
        f.close()
    
    def test_global_metadata_accessible(self, simple_brim_file):
        """Test that global metadata is accessible from data group."""
        f = brim.File(simple_brim_file, mode='r+')
        data = f.get_data()
        md = data.get_metadata()
        
        # Global metadata should be accessible
        wavelength = md['Optics.Wavelength']
        assert wavelength is not None
        f.close()
