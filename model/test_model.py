from model_loader import ChestXrayModel

image_path = "sample_xray.jpg"

model = ChestXrayModel()
results = model.predict(image_path)

print("Prediction Results:")
for label, prob in results:
    print(f"{label}: {prob:.2f}")
from summarizer import MedicalSummarizer

# Initialize summarizer
summarizer = MedicalSummarizer()

# Get top 5 predictions for summarization
top_conditions = results[:5]  # Already sorted

# Generate and print AI summary
summary = summarizer.generate_summary(top_conditions)
print("\nAI Medical Summary:\n", summary)

