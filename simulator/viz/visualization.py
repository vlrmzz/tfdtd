import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def viz2D(sim,field):

    #pmlcolor = (0, 0, 0, 0.1)
    #pml_thickness = sim.pml_thickness

    # plot the Ez field with cmap jet
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.transpose(field), cmap='Blues', extent=[0, sim.nx*sim.dx*1e6, sim.ny*sim.dy*1e6, 0])
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_title('Field')

    # plot the geometry mask with cmap binary
    ax.imshow(np.transpose(sim.eps_r), cmap='binary', extent=[0, sim.nx*sim.dx*1e6, sim.ny*sim.dy*1e6, 0],alpha=0.3)

    fig.colorbar(im)
    plt.show()

# # # Sample data for the energy distribution
# # #tm_sim = lambda: None  # Dummy object to simulate your data
# # #tm_sim.Dz = np.random.rand(100, 100)  # Replace with your actual energy data
# # tm_sim.nx, tm_sim.ny = tm_sim.Dz.shape

# # # Properties of the energy sources, geometry, and PML layers
# # source_positions = [(10, 10), (90, 90)]  # Replace with actual source positions
# # detector_positions = [(90, 10), (10, 90)]  # Replace with actual detector positions
# # pml_thickness = tm_sim.pml_thickness

# # # Colors and plot settings
# # cmap = "Blues"
# # geom_cmap = "jet"
# # pbcolor = "C3"
# # pmlcolor = (0, 0, 0, 0.1)
# # objcolor = (1, 0, 0, 0.1)
# # srccolor = "C0"
# # detcolor = "C2"

# # # Create the right legend entries
# # fig, ax = plt.subplots()
# # plt.plot([], lw=7, color=objcolor, label="Objects")
# # plt.plot([], lw=7, color=pmlcolor, label="PML")
# # plt.plot([], lw=3, color=pbcolor, label="Periodic Boundaries")
# # plt.plot([], lw=3, color=srccolor, label="Sources")
# # plt.plot([], lw=3, color=detcolor, label="Detectors")

# # # Display the geometry
# # #plt.imshow(tm_sim.gaz, cmap=geom_cmap, extent=[0, tm_sim.nx, 0, tm_sim.ny], alpha=1.0)

# # # Display the energy distribution
# # plt.imshow(tm_sim.Dz, cmap=cmap, extent=[0, tm_sim.nx, 0, tm_sim.ny], alpha=1.0)

# # # Draw PML layers
# # for edge in [0, tm_sim.nx, 0, tm_sim.ny]:
# #     plt.axhline(edge, lw=pml_thickness, color=pmlcolor)
# #     plt.axvline(edge, lw=pml_thickness, color=pmlcolor)

# # # Draw sources
# # ##for src_x, src_y in source_positions:
# # ##    plt.plot(src_x, src_y, 'o', color=srccolor)

# # # Draw detectors
# # ##for det_x, det_y in detector_positions:
# # ##   plt.plot(det_x, det_y, 'x', color=detcolor)

# # # Configure axis and colorbar
# # ax.set_xticks(np.arange(0, tm_sim.nx, step=20))
# # ax.set_yticks(np.arange(0, tm_sim.ny, step=20))
# # plt.colorbar(cmap='jet')
# # plt.xlabel('cm')
# # plt.ylabel('cm')

# # # Display the legend and show the plot
# # #plt.legend()
# # plt.show()

def viz_detector(simulation,detector):
    # Define simulation parameters
    time_steps = simulation.time_steps
    dt = simulation.dt

    # Create a simple FDTD detector output (replace this with your actual detector data)
    t = np.arange(0, time_steps * dt, dt)
    detector_output = np.stack([t.numpy() for t in detector.recorded_values]).squeeze()

    # Compute the Fourier Transform of the detector output
    detector_fft = np.fft.fft(detector_output)
    freqs = np.fft.fftfreq(time_steps, dt)

    # Calculate the magnitude and phase of the Fourier Transform
    magnitude = np.abs(detector_fft)
    phase = np.angle(detector_fft)

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # Plot the time-domain detector output
    axs[0, 0].plot(t, detector_output)
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Amplitude')
    axs[0, 0].set_title('Detector Output (Time Domain)')

    # Plot the magnitude of the Fourier Transform
    axs[0, 1].plot(freqs[:time_steps // 2][0:100], magnitude[:time_steps // 2][0:100])
    axs[0, 1].set_xlabel('Frequency (Hz)')
    axs[0, 1].set_ylabel('Magnitude')
    axs[0, 1].set_title('Detector Output FFT (Magnitude)')
    max_xticks = 10
    axs[0, 1].xaxis.set_major_locator(MaxNLocator(max_xticks))

    # Plot the phase of the Fourier Transform
    axs[1, 0].plot(freqs[:time_steps // 2], phase[:time_steps // 2])
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('Phase (radians)')
    axs[1, 0].set_title('Detector Output FFT (Phase)')

    # Remove the unused subplot
    axs[1, 1].axis('off')

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the figure
    plt.show()