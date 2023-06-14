import os
import re
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

    def __init__(self, config_file, projects_dir, project_dir):
        """
        The constructor for the `Project` class.

        Parameters:
            config_file (str): The path to the directory containing the configuration file.
            projects_dir (str): The path to the directory where all projects are saved.
            project_dir (str): The path to the main directory for the project.
        """
        
        self.projects_dir = projects_dir
        self.project_dir = os.path.join(projects_dir, project_dir)
        self.simulation = None
        self.config_file = config_file

        # Check if the projects directory exists, if not, create it
        os.makedirs(self.projects_dir, exist_ok=True)

    def copy_config(self):
        """
        Copies the config file to the target directory.

        The target directory is the 'config' subdirectory within the current simulation directory.

        Returns
        -------
        None
        """

        # Path to the 'config' subdirectory within the current simulation directory
        config_dir = os.path.join(self.sim_dir, 'config')

        # Copy the config file to the target directory
        shutil.copy2(self.config_file, config_dir)


    def setup_directories(self):
        """
        Sets up the necessary directories for a new simulation.

        This function first checks whether the main project directory exists. If it doesn't, the function creates it.
        It then scans the project directory for any existing simulation directories. These are expected to follow the
        naming convention "simulation" followed by an integer (e.g., "simulation1", "simulation2", etc.).
        
        The function determines the highest existing simulation number and creates a new simulation directory
        with a number one higher than the highest existing number.
        
        Finally, the function creates the necessary subdirectories ('detectors', 'geometry', 'fields', 'config')
        within the new simulation directory.
        
        Parameters
        ----------
        self : object
            The object instance. Should have a `project_dir` attribute indicating the path to the project directory.

        Returns
        -------
        None
        """
        
        # Create the main project directory if it doesn't exist
        os.makedirs(self.project_dir, exist_ok=True)

        # Find all existing simulation directories
        sim_dirs = [d for d in os.listdir(self.project_dir) if re.match(r'simulation\d+', d)]

        # Extract the simulation numbers and find the highest one
        sim_nums = [int(re.search(r'\d+', d).group()) for d in sim_dirs]
        max_sim_num = max(sim_nums) if sim_nums else 0

        # Create a new simulation directory with a number one higher than the highest number
        self.sim_dir = os.path.join(self.project_dir, f'simulation{max_sim_num + 1}')
        os.makedirs(self.sim_dir, exist_ok=True)

        # Create subdirectories
        subdirs = ['detectors', 'geometry', 'fields', 'config']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.sim_dir, subdir), exist_ok=True)
            
        # Copy configuration file to the 'config' subdirectory
        self.copy_config()

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
        self.setup_simulation()
        
    def run(self):
        self.simulation.run()   

