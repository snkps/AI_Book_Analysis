# app.py
from flask import Flask, render_template, request
import pandas as pd
from model_service import analyze_emotions, summarize_text_bertemb

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    annotation = ""
    overall_emotion = ""
    sentence_emotions_df = pd.DataFrame(columns=["Sentence", "Emotion"])
    counts_md = ""
    text_input = ""

    do_annotation = True
    do_emotions = True

    if request.method == "POST":
        text_input = request.form.get("input_text", "")
        do_annotation = request.form.get("do_annotation") == "on"
        do_emotions = request.form.get("do_emotions") == "on"

        if text_input.strip():
            if do_annotation:
                annotation = summarize_text_bertemb(text_input)

            if do_emotions:
                sentences, emotions, overall, counts = analyze_emotions(text_input)
                overall_emotion = overall
                sentence_emotions_df = pd.DataFrame({"Sentence": sentences, "Emotion": emotions})
                total_count = sum(counts.values())
                counts_md = [f"{k}: {v} ({(v/total_count*100):.1f}%)" for k,v in counts.items()]

    return render_template(
        "index.html",
        input_text=text_input,
        annotation=annotation,
        overall_emotion=overall_emotion,
        sentence_emotions=sentence_emotions_df.to_dict(orient="records"),
        counts_md=counts_md,
        do_annotation=do_annotation,
        do_emotions=do_emotions
    )

if __name__ == "__main__":
    app.run(debug=True)
