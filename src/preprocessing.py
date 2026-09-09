import ssl
import urllib.request
from pathlib import Path
import mne
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf

def download_edf_file(subject_id: int, run_id: int) -> Path:
    """
    Downloads an EDF file directly from PhysioNet, explicitly overriding
    macOS SSL certificate checks.
    """
    sub_str = f"S{subject_id:03d}"
    run_str = f"{sub_str}R{run_id:02d}.edf"
    
    # Match standard MNE cache directory structure
    base_dir = Path.home() / "mne_data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / sub_str
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / run_str
    
    if not file_path.exists():
        url = f"https://physionet.org/files/eegmmidb/1.0.0/{sub_str}/{run_str}"
        print(f"Downloading {run_str}...")
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response, open(file_path, "wb") as out_file:
            out_file.write(response.read())
            
    return file_path

# Standard 21-channel motor cortex layout surrounding C3, Cz, and C4
MOTOR_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
]

def preprocess_subject(
    subject_id: int, 
    runs: list = [4, 8, 12], 
    l_freq: float = 8.0, 
    h_freq: float = 30.0, 
    tmin: float = 0.5, 
    tmax: float = 3.5,
    motor_only: bool = False
) -> mne.Epochs:
    raw_fnames = [str(download_edf_file(subject_id, r)) for r in runs]
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = concatenate_raws(raws)
    
    if hasattr(eegbci, 'standardize'):
        eegbci.standardize(raw)
    else:
        raw.rename_channels(lambda x: x.strip('.').strip())
        
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False)
    
    # Restrict to motor cortex subset if requested
    if motor_only:
        raw.pick_channels([ch for ch in MOTOR_CHANNELS if ch in raw.ch_names])
        
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    target_event_id = {k: v for k, v in event_id.items() if k in ['T1', 'T2']}
    
    epochs = mne.Epochs(
        raw, 
        events=events, 
        event_id=target_event_id, 
        tmin=tmin, 
        tmax=tmax, 
        baseline=None, 
        preload=True, 
        verbose=False
    )
    
    return epochs