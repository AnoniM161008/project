async function analyzeSentiment() {
    const text = document.getElementById('textInput').value;

    if (!text) {
        alert("Please enter some text!");
        return;
    }

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        console.log(data);  // Проверьте, что возвращает сервер

        // Обновляем результаты на странице
        document.getElementById('nltkResult').innerText = data.nltk_sentiment;
        document.getElementById('vaderResult').innerText = data.vader_sentiment;
        document.getElementById('vaderScore').innerText = JSON.stringify(data.vader_score);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while analyzing the text.');
    }
}