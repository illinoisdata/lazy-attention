# e2e test for batch processing using RAGEngine

############################################################################################################
# TODO:
# test when all seqs fit in one batch
# test when no enough memory to fit all seqs to one batch
# test when a new seq come with the docs used before by a finished previous seq.(async scenerio)

# for e2e test, use VLLM as reference implementation, test if the ModelInputForGPUWithSamplingMetadata is the same.
# also, using the original drag impl as reference, test if the generated text is the same.
############################################################################################################