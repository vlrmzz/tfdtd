import os
import shutil
import logging

from simulator.base import TFDTD2D
from simulator.fdtd import FDTD2D
import yaml
from simulator.sources import LineSource, PointSource
from simulator.detectors import LineDetector, PointDetector
from simulator.utilities import print_setup_info 

logger = logging.getLogger(__name__)

class Project:
    """
    The `Project` class is used to manage a simulation project. It provides methods for setting up project directories, 
    copying configuration files, setting up the simulation, and running the simulation.

    Attributes:
        config_folder (str): The path to the directory containing the configuration file.
        project_dir (str): The path to the main directory for the project.
        simulation (TFDTD2D): The simulation object, which is created when `setup_simulation` is called.
        config_file (str): The path to the configuration file.
    """

    def __init__(self, config_folder, project_dir):
        """
        The constructor for the `Project` class.

        Parameters:
            config_folder (str): The path to the directory containing the configuration file.
            project_dir (str): The path to the main directory for the project.
        """
        self.config_folder = config_folder
        self.project_dir = project_dir
        self.simulation = None
        self.config_file = config_folder
    def setup_directories(self):
        # Create the main project directory if it doesn't exist
        os.makedirs(self.project_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ['detectors', 'geometry', 'fields', 'config']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.project_dir, subdir), exist_ok=True)
            
    def copy_config(self):

        # Copy the config file to the target directory
        shutil.copy2(self.config_file, os.path.join(self.project_dir, 'config'))

    def setup_simulation(self):
        # Read the YAML configuration file
        with open(self.config_file, 'r') as file:
            config = yaml.safe_load(file)

        # Create a list to hold the sources
        sources = []

        # Loop through the sources in the configuration
        for source_config in config['sources']:
            # Create the appropriate source object based on the type
            if source_config['type'] == 'line':
                source = LineSource(source_config)
            elif source_config['type'] == 'point':
                source = PointSource(source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_config['type']}")
            
            # Add the source to the list
            sources.append(source)
        
        # Create a list to hold the detectors
        detectors = []
         
        for detector_config in config['sources']:
            # Create the appropriate source object based on the type
            if detector_config['type'] == 'line':
                detector = LineDetector(detector_config)
            elif detector_config['type'] == 'point':
                detector = PointDetector(detector_config)
            else:
                raise ValueError(f"Unsupported source type: {detector_config['type']}")
            
            # Add the source to the list
            detectors.append(detector)
        
        self.simulation = TFDTD2D(config_file=self.config_file)
        for source in sources:
            self.simulation.add_source(source)
        for detector in detectors:
            self.simulation.add_detector(detector)
            
        logger.info(print_setup_info(self.simulation))

    def setup(self):
        self.setup_directories()
        self.copy_config()
        self.setup_simulation()
        
    def run(self):
        self.simulation.run()   

