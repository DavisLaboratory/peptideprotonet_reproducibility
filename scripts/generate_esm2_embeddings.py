from pipp import ESM2
from typing import List, cast
import pickle
from argparse import ArgumentParser
import os


def compute_esm2_embeddings(input_path: str, model_name: str, embedding_layer: int):
    model = ESM2(model_name)

    with open(input_path, "rb") as f:
        data = pickle.load(f)

    sequences = cast(List[str], list(data.values()))
    embeddings = model.get_embeddings(sequences, batch_size=512)

    precursor_embedding = dict(zip(data.keys(), embeddings))

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"embeddings_{model_name}_l{embedding_layer}.pkl"
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(precursor_embedding, f)

    print("Embeddings generated and saved to", filename)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the input file",
        default="outputs/prototype_sequences.pkl",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        help="ESM model name",
        default="esm2_t6_8M_UR50D",
    )
    parser.add_argument(
        "--representation_layer",
        type=int,
        help="Layer to extract embeddings from",
        default=6,
    )

    args = parser.parse_args()

    compute_esm2_embeddings(args.input_path, args.esm_model, args.representation_layer)
