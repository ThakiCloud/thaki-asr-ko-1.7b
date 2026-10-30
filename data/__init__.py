"""Data preparation for SFT — audio normalisation + Qwen3-ASR SFT format.

Clean, generic cut: it documents and reproduces the exact audio/preprocessing
contract the released model was trained with (16 kHz mono PCM_16, the
`language Korean<asr_text>...` SFT text format, and the eval-contamination leak
gate), with no reference to any internal storage or training corpus.
"""
