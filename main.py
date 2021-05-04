import argparse
import textwrap

import datasets
import models
import trainer


def main(args):
    model = models.get_model(args.model, args.dataset)

    dataset = model.dataset
    train, test = datasets.load_data(dataset)
    train, validation = datasets.split_data(train, fractions=[0.8, 0.2])

    trainer.train(model, train, validation, args.epochs, args.batch_size,
                  args.learning_rate,
                  dataset_update=args.dataset_update, increments=args.splits,
                  use_ewc=args.ewc, ewc_lambda=args.ewc_lambda,
                  ewc_samples=args.ewc_samples,
                  use_fim=args.fim, fim_threshold=args.fim_threshold,
                  fim_samples=args.fim_samples,
                  use_incdet=args.incdet,
                  incdet_threshold=args.incdet_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
    Incremental learning demo.
    
    Train a model on a variable dataset. Dataset update options are:    
        1. full: Use the full dataset for the whole training process.
        2. increment: Start with a subset of classes and periodically add more.
        3. switch: Start with a subset of classes and periodically use a 
                   disjoint subset.
        4. permute: Start with the full dataset, and add permutations of the
                    pixels. Permutations are shared for each split.
    
    `splits` controls how many subsets to split the dataset into, with the given
    `epochs` being distributed evenly across them."""),
        formatter_class=argparse.RawTextHelpFormatter
    )

    training = parser.add_argument_group(title="Normal training")
    training.add_argument("--batch-size", type=int, default=256, metavar="N",
                          help="Number of inputs to process simultaneously")
    training.add_argument("--epochs", type=int, default=30, metavar="N",
                          help="Number of iterations through training dataset")
    training.add_argument("--learning-rate", type=float, default=0.001,
                          metavar="LR", help="Initial learning rate for Adam")
    training.add_argument("--model", type=str, choices=models.models(),
                          default="mlp", help="Neural network to train")
    training.add_argument("--dataset", type=str, choices=datasets.datasets(),
                          help="Dataset to use (default: specified by model)")

    inc = parser.add_argument_group(title="Incremental learning")
    inc.add_argument("--dataset-update", type=str,
                     choices=trainer.increment_options(), default="full",
                     help="How to change the dataset during training")
    inc.add_argument("--splits", type=int, default=5, metavar="N",
                     help="Number of dataset partitions (equally sized)")

    ewc = parser.add_argument_group(title="Elastic weight consolidation")
    ewc.add_argument("--ewc", action="store_true",
                     help="Use EWC to preserve accuracy on previous datasets")
    ewc.add_argument("--ewc-lambda", type=float, default=0.1, metavar="L",
                     help="Relative importance of old tasks vs new tasks")
    ewc.add_argument("--ewc-samples", type=int, default=100, metavar="N",
                     help="Inputs to sample to estimate weight importance")

    fim = parser.add_argument_group(title="Fisher information masking")
    fim.add_argument("--fim", action="store_true",
                     help="Use FIM to preserve accuracy on previous datasets")
    fim.add_argument("--fim-threshold", type=float, default=1e-6, metavar="T",
                     help="Threshold controlling when to freeze weights")
    fim.add_argument("--fim-samples", type=int, default=100, metavar="N",
                     help="Inputs to sample to estimate weight importance")

    incdet = parser.add_argument_group(title="Incremental detection")
    incdet.add_argument("--incdet", action="store_true",
                        help="Use IncDet to address EWC's limitations")
    incdet.add_argument("--incdet-threshold", type=float, default=1e-6,
                        metavar="B",
                        help="Threshold for IncDet gradient clipping")

    flags = parser.parse_args()

    # Fix up some nonsensical argument combinations.
    if flags.dataset_update == "full":
        flags.splits = 1

    try:
        main(flags)
    except KeyboardInterrupt:
        pass  # Just want to suppress the stack trace
