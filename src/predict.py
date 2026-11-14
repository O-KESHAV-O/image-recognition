import argparse
from classifier import ImageClassifier

def main():
    parser = argparse.ArgumentParser(description="Image Recognition CLI")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to image")
    args = parser.parse_args()

    classifier = ImageClassifier()
    predictions = classifier.predict(args.image)
    print(predictions)

if __name__ == "__main__":
    main()
