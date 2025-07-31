# model/summarizer.py

class MedicalSummarizer:
    def __init__(self):
        # Realistic summaries for actual medical conditions
        self.condition_explanations = {
            "Cardiomegaly": {
                "overview": "Cardiomegaly refers to an enlarged heart, which may indicate chronic high blood pressure or heart disease.",
                "next_steps": "A follow-up with a cardiologist is recommended. An echocardiogram can assess heart size and function.",
                "insight": "Early management with medications and lifestyle adjustments can help prevent complications like heart failure."
            },
            "Pneumonia": {
                "overview": "Pneumonia is an infection that causes inflammation and fluid buildup in the lungs.",
                "next_steps": "Medical consultation is advised for possible antibiotics or antivirals. A chest CT may be used for confirmation.",
                "insight": "Prompt treatment typically leads to full recovery. It’s especially important to monitor in elderly patients."
            },
            "Pleural Effusion": {
                "overview": "Pleural effusion is the accumulation of fluid around the lungs, often due to infection, heart issues, or cancer.",
                "next_steps": "Ultrasound or CT imaging followed by fluid drainage may be necessary. Identifying the cause is critical.",
                "insight": "Most effusions are manageable, but require monitoring to prevent recurrence or respiratory distress."
            },
            "No Finding": {
                "overview": "No abnormalities were detected in the chest X-ray. Lungs and heart appear within normal limits.",
                "next_steps": "No follow-up is needed unless clinical symptoms persist or worsen.",
                "insight": "A clear result is reassuring. Maintain regular checkups and a healthy lifestyle."
            },
            "Atelectasis": {
                "overview": "Atelectasis is a partial collapse of lung tissue, reducing oxygen exchange in the affected area.",
                "next_steps": "Breathing exercises, physical therapy, and addressing underlying causes (like mucus plug or tumor) are key.",
                "insight": "It is often reversible, especially if caught early. Monitoring and support can aid full recovery."
            },
            "Edema": {
                "overview": "Pulmonary edema means fluid accumulation in the lungs, commonly from heart failure or severe infections.",
                "next_steps": "Urgent medical evaluation is required. Diuretics and oxygen therapy are typical treatments.",
                "insight": "If treated promptly, symptoms can improve quickly. Chronic cases need heart function monitoring."
            },
            # Add more if needed
        }

    def summarize(self, predictions):
     if not predictions:
        return "✅ No conditions detected. The chest X-ray appears normal."

     # Remove "No Finding" if other confident conditions exist
     has_significant_findings = any(
     condition != "No Finding" and confidence > 0.5 for condition, confidence in predictions
     )

     filtered_preds = [
        (condition, confidence)
        for condition, confidence in predictions
        if not (condition == "No Finding" and has_significant_findings)
     ]
     filtered_preds.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence

     summaries = []

     for condition, confidence in filtered_preds:
        if condition not in self.condition_explanations:
            continue  # Skip unknown conditions, no bluff

        info = self.condition_explanations[condition]

        summary = f"""🔍 **{condition}** ({confidence * 100:.1f}% confidence)

     Condition Overview: 
     {info['overview']}

    What to Do Next:  
    {info['next_steps']}

    Health Insight:  
    {info['insight']}
   """
        summaries.append(summary)

     return "\n\n---\n\n".join(summaries)
