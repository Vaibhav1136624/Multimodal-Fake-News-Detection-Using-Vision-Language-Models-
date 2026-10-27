# Initial public benchmark: Fakeddit

Official release: https://github.com/entitize/Fakeddit
Paper: https://aclanthology.org/2020.lrec-1.755/

Use the official multimodal-only release and clean_title text. Start with a bounded
sample after downloading metadata, then acquire only its images. Record the release,
sample seed, original IDs, original split, original labels, and missing-image counts.
Verify label semantics against the paper before converting into this project's
0=real, 1=fake convention. Do not infer polarity from the numeric label alone.

Fakeddit uses distant supervision, not individually fact-checked truth for each item.
Its label categories include satire and misleading associations. Describe experiments
as benchmark classification, not proof that factual claims have been verified.

Group choice is an unresolved methodological step: derive story/duplicate groups
without using target labels; do not use unique post IDs as MERF groups. Subreddit
can be strongly associated with label construction, so it is not a neutral default
for grouping. Evaluate the official split separately for literature comparisons;
the current group-held-out split is a distinct experiment.

Check release terms and cite the authors. Competition rules are separate from this
binary research experiment. No dataset has been downloaded or processed yet.
