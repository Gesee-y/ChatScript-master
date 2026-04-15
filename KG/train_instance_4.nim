import std/os
import Training

when isMainModule:
  let args = @[
    "--train", "KG" / "DATA" / "train_instance_4_train_pruned.txt",
    "--answers", "KG" / "DATA" / "train_instance_4_answers.csv",
    "--outdir", "KG" / "artifacts" / "instance_4",
    "--epochs", "200",
    "--batch", "32",
    "--eval", "5",
    "--seed", "42",
    "--refine-topk", "20"
  ]
  runTrainingPipeline(args)
