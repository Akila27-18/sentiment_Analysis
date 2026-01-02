function analyze() {
    const text = document.getElementById("textInput").value;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Sentiment: " + data.sentiment.toUpperCase();
    });
}
