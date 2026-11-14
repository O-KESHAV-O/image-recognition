import argparse
from classifier import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description="Image Recognition CLI")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to image")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions")
    args = parser.parse_args()

    classifier = ImageClassifier()
    predictions = classifier.predict(args.image, top_k=args.top_k)
    print("\n=== Top Predictions ===")
    for item in predictions:
        print(f"[{item['rank']}] {item['label']:<25} : {item['confidence']*100:6.2f}%")
    print("=======================\n")

if __name__ == "__main__":
    main()
