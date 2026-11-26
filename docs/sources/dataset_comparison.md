# Local dataset comparison

Facts established from local inspection and the project's existing audit. Dataset-native file counts do not represent independent recording counts.

| Property | Al-Emadi binary corpus | Drone-detection-dataset audio | Salford local folder |
|---|---:|---:|---:|
| WAV files | 11,704 | 90 | 174 |
| Label/file categories | 1,332 yes_drone; 10,372 unknown | 30 DRONE; 30 HELICOPTER; 30 BACKGROUND | 165 event-style files; 9 calibration files |
| Sample rate | 16,000 Hz | 44,100 Hz | 50,000 Hz |
| Channels | Mono | Stereo | Mono |
| Encoding | PCM16 | PCM16 | Float32 WAV |
| Duration | Mostly approximately 1 s | Approximately 10 s; aggregate 900.0031 s | 11.88394–30 s, median 20 s |
| Measured WAV bytes | 384,021,184 | 158,764,512 | 742,054,912 |

Al-Emadi's README describes indoor propeller recordings and random-noise augmentation; unknown audio includes ESC-50, Speech Commands-related noise and author-created silence. The binary and multiclass trees duplicate the task's audio content and are preserved as downloaded. The existing project uses only binary membership.

The thesis dataset also contains 650 videos (365 IR, 285 visible), MATLAB labels and Video_dataset_description.xlsx. Its README reports 203,328 annotated frames. Video classes are airplane, bird, drone and helicopter. The published dataset paper describes acquisition at three Swedish airports; video source cells also identify some visible YouTube material.

Salford's local folder contains DNM_SQM_FLYO_AESCT.xlsx and no README. The workbook contains 14 numbered flyover rows, microphone M1–M9 acoustic metrics, flight metadata and summary sheets. Filenames use 3p, Fp, M3 and Yn codes. The 165 event files include 18 each for M1–M4/M6–M9 and 21 for M5. The official record describes field recordings at Edzell, Scotland, on 17 August 2022. A file count is not a count of independent flights.

Source files, licenses, headers and checksum inventories are preserved with their source revision. Exact upstream Git commits and the downloaded Salford published version have not been established from the folder names. These uncertainties remain explicit.

References: https://github.com/saraalemadi/DroneAudioDataset ; https://github.com/DroneDetectionThesis/Drone-detection-dataset ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8573135/ ; https://doi.org/10.17866/rd.salford.22133411
