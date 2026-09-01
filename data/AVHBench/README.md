# AVHBench annotation index

This directory is the local AVHBench copy used by the OWP paper. It contains
`QA.json` with 6,408 annotations, 2,327 video files, and the matching 2,327
audio files. The OWP comparison uses the 5,302 non-caption questions: 1,136
audio-driven visual hallucination questions, 2,290 video-driven audio
hallucination questions, and 1,876 audiovisual matching questions.

The source release retains the annotations but does not track the media. The
official AVHBench README currently distributes its media through a Google Drive
subset. For automatic preparation, publish an equivalent copy in a Hugging Face
dataset repository and set `OWP_AVHBENCH_DATASET_ID`; OWP then downloads and
unpacks it into the user cache. Please preserve the dataset's original citation
and usage terms.
