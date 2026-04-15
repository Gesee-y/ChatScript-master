import std/os
import Training

when isMainModule:
  let args = @[
    "--train", "KG" / "DATA" / "train_instance_1_train_pruned.txt",
    "--answers", "KG" / "DATA" / "train_instance_1_answers.csv",
    "--outdir", "KG" / "artifacts" / "instance_1",
    "--epochs", "200",
    "--batch", "32",
    "--eval", "5",
    "--seed", "42",
    "--refine-topk", "20"
  ]
  runTrainingPipeline(args)
