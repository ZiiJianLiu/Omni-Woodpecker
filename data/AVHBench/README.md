# AVHBench annotation index

This directory contains the AVHBench annotation index used by OWP. `QA.json`
has 6,408 annotations. The source release does not track the corresponding
media; callers provide local files or configure an equivalent Hugging Face
dataset repository. The OWP comparison uses the 5,302 non-caption questions: 1,136
audio-driven visual hallucination questions, 2,290 video-driven audio
hallucination questions, and 1,876 audiovisual matching questions.

The source release retains the annotations but does not track the media. The
official AVHBench README currently distributes its media through a Google Drive
subset. For automatic preparation, publish an equivalent copy in a Hugging Face
dataset repository and set `OWP_AVHBENCH_DATASET_ID`; OWP then downloads and
unpacks it into the user cache. Please preserve the dataset's original citation
and usage terms.
