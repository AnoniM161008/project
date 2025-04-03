document.getElementById('analyzeButton').addEventListener('click', async () => {
    const text = document.getElementById('textInput').value;

    try {
        const response = await fetch('http://localhost:3000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json; charset' +
                    '=utf-8',
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        document.getElementById('result').innerHTML =
            <h2>Результат:</h2>
            <p>Тональность: <span style="color: ${data.polarity > 0 ? 'green' : 'red'}">${data.sentiment}</span></p>
        ;
    } catch (error) {
        console.error("Ошибка:", error);
        alert("Не удалось проанализировать текст");
    }
});