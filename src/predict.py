import argparse
from classifier import ImageClassifier

DEFAULT_SAMPLE_IMAGE = "https://images.unsplash.com/photo-1552053831-71594a27632d?auto=format&fit=crop&w=800&q=80"

def main():
    parser = argparse.ArgumentParser(description="Image Recognition CLI")
    parser.add_argument("--image", type=str, default=None, help="Path or URL to image (optional)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()

    image_source = args.image or DEFAULT_SAMPLE_IMAGE
    classifier = ImageClassifier()
    predictions = classifier.predict(image_source, top_k=args.top_k)
    print(f"\nTarget: {image_source}")
    print("\n=== Top Predictions ===")
    for item in predictions:
        print(f"[{item['rank']}] {item['label']:<25} : {item['confidence']*100:6.2f}%")
    print("=======================\n")

if __name__ == "__main__":
    main()
