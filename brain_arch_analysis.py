import numpy as np
import torch
import torch.nn as nn
import gradio as gr
import plotly.graph_objects as go
import mne
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- NOTE: Re-using the core classes and feature extraction from our last script ---
# (Encoder, Predictor, WorldModel, PairedEEGDataset, create_eeg_features, etc.)
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, latent_dim))
    def forward(self, x): return self.net(x)
class Predictor(nn.Module):
    def __init__(self, latent_dim=32, depth=3):
        super().__init__()
        layers = [nn.Linear(latent_dim, 128), nn.ReLU()]
        for _ in range(depth - 1): layers.extend([nn.Linear(128, 128), nn.ReLU()])
        layers.append(nn.Linear(128, latent_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)
class WorldModel(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.predictor = Predictor(latent_dim)
EEG_REGIONS = {"All": [], "Occipital": ['O1', 'O2', 'OZ', 'POZ', 'PO3', 'PO4', 'PO7', 'PO8'], "Temporal": ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8'], "Parietal": ['P1', 'P2', 'P3', 'P4', 'PZ', 'CP1', 'CP2'], "Frontal": ['FP1', 'FP2', 'FZ', 'F1', 'F2', 'F3', 'F4'], "Central": ['C1', 'C2', 'C3', 'C4', 'CZ', 'FC1', 'FC2']}

def create_eeg_features(edf_file, region="All"):
    frequency_bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    raw.rename_channels(lambda name: name.strip().replace('.', '').upper())
    
    if region != "All":
        region_channels = EEG_REGIONS[region]
        available_channels = [ch for ch in region_channels if ch in raw.ch_names]
        if not available_channels: return np.array([]), 0, []
        raw.pick_channels(available_channels)
    
    fs = 100.0
    raw.resample(fs, verbose=False)
    
    band_filtered_data = {band: raw.copy().filter(l_freq=low, h_freq=high, fir_design='firwin', verbose=False).get_data() for band, (low, high) in frequency_bands.items()}
    samples_per_epoch = int(0.5 * fs)
    all_epochs_features = []
    
    for i in range(0, raw.n_times - samples_per_epoch, samples_per_epoch):
        epoch_band_powers = [np.log1p(np.mean(band_filtered_data[band][:, i:i+samples_per_epoch]**2, axis=1)) for band in frequency_bands.keys()]
        all_epochs_features.append(np.stack(epoch_band_powers, axis=1))
        
    all_epochs_features = np.array(all_epochs_features)
    
    # Create feature names for visualization
    feature_names = [f"{ch}-{band}" for ch in raw.ch_names for band in frequency_bands.keys()]
    
    n_epochs, n_channels, n_bands = all_epochs_features.shape
    flattened_features = all_epochs_features.reshape(n_epochs, n_channels * n_bands)
    
    mean = np.mean(flattened_features, axis=0, keepdims=True)
    std = np.std(flattened_features, axis=0, keepdims=True)
    std[std == 0] = 1
    normalized_features = (flattened_features - mean) / std
    
    return normalized_features, feature_names

# ==================================
# NEW: Architectural Analysis Functions
# ==================================

def create_correlation_fingerprint(feature_data, feature_names):
    """Visualizes the correlation between all input features."""
    if feature_data.shape[0] < 2: return go.Figure()
    
    correlation_matrix = np.corrcoef(feature_data.T)
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=feature_names,
        y=feature_names,
        colorscale='RdBu_r',
        zmid=0
    ))
    fig.update_layout(
        title="Functional Connectivity Fingerprint (All Features)",
        template='plotly_dark',
        height=600,
        yaxis=dict(autorange='reversed')
    )
    return fig

def create_state_flow_diagram(latent_trajectory, n_states=5):
    """Identifies stable states and visualizes transitions between them."""
    if len(latent_trajectory) < n_states: return go.Figure()

    # 1. Find stable states using K-Means clustering
    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    state_labels = kmeans.fit_predict(latent_trajectory)
    state_centers = kmeans.cluster_centers_
    
    # 2. Count transitions between states
    transitions = np.zeros((n_states, n_states))
    for i in range(len(state_labels) - 1):
        start_state = state_labels[i]
        end_state = state_labels[i+1]
        if start_state != end_state:
            transitions[start_state, end_state] += 1
            
    # 3. Create the flow diagram (Sankey plot)
    source_nodes, target_nodes, values = [], [], []
    labels = [f"State {i}" for i in range(n_states)]
    for i in range(n_states):
        for j in range(n_states):
            if transitions[i, j] > 0:
                source_nodes.append(i)
                target_nodes.append(j)
                values.append(transitions[i, j])

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
        link=dict(source=source_nodes, target=target_nodes, value=values)
    )])
    fig.update_layout(title_text="Map of Thoughts: State-Space Flow Diagram", template='plotly_dark', height=600)
    return fig

# ==================================
# Gradio App
# ==================================

def run_analysis(edf_file, region, latent_dim, epochs, progress=gr.Progress()):
    # This function now trains the model and then performs the architectural analysis
    if edf_file is None: raise gr.Error("Please upload an EEG file.")
    
    progress(0.1, desc="Extracting features...")
    feature_data, feature_names = create_eeg_features(edf_file.name, region)
    if len(feature_data) == 0: raise gr.Error("Could not extract features for the selected region.")
    
    input_dim = feature_data.shape[1]
    
    # Dummy training for demonstration (in a real app, this would be the full WorldModel)
    progress(0.5, desc="Generating trajectory (simulating trained model)...")
    pca = PCA(n_components=latent_dim)
    latent_trajectory = pca.fit_transform(feature_data) # Use PCA as a proxy for the encoder

    # Generate the new visualizations
    progress(0.8, desc="Analyzing architecture...")
    fig_corr = create_correlation_fingerprint(feature_data, feature_names)
    fig_flow = create_state_flow_diagram(latent_trajectory)
    
    return fig_corr, fig_flow

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Brain Architecture Analyzer")
    gr.Markdown("Visualize the entire multi-dimensional system as a cohesive whole, revealing its functional architecture and 'map of thoughts'.")
    
    with gr.Row():
        edf_input = gr.File(label="Upload EEG File (.edf)")
        region_selector = gr.Dropdown(choices=list(EEG_REGIONS.keys()), value="Occipital", label="Select Brain Region")
        run_button = gr.Button("Analyze Brain Architecture", variant="primary")
        
    with gr.Tab("Functional Connectivity Fingerprint"):
        corr_plot = gr.Plot()
        
    with gr.Tab("Map of Thoughts"):
        flow_plot = gr.Plot()
        
    run_button.click(
        fn=run_analysis,
        inputs=[edf_input, region_selector, gr.Slider(2, 64, value=8, visible=False), gr.Slider(10, 200, value=50, visible=False)],
        outputs=[corr_plot, flow_plot]
    )

if __name__ == "__main__":
    app.launch(debug=True)