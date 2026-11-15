import argparse
import sys
from classifier import ImageClassifier


# Default sample image URL (Golden Retriever)
DEFAULT_SAMPLE_IMAGE = "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=800&q=80"


def main():
    parser = argparse.ArgumentParser(description="Image Recognition CLI")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an image file or a web URL (defaults to sample image if omitted)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to display (default: 5)",
    )
    args = parser.parse_args()

    image_source = args.image or DEFAULT_SAMPLE_IMAGE

    try:
        classifier = ImageClassifier()
        predictions = classifier.predict(image_source, top_k=args.top_k)

        print(f"\nTarget: {image_source}")
        print("\n=== Top Predictions ===")
        for item in predictions:
            confidence_pct = item["confidence"] * 100
            print(f"[{item['rank']}] {item['label']:<25} : {confidence_pct:6.2f}%")
        print("=======================\n")

    except Exception as exc:
        print(f"Error during classification: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

