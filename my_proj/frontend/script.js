document.getElementById('analyzeButton').addEventListener('click', function() {
    const text = document.getElementById('textInput').value;
    const sentiment = analyzeSentiment(text);
    displayResult(sentiment);
});

function analyzeSentiment(text) {
    // Пример простого анализа тональности
    const positiveWords = ['хорошо', 'отлично', 'прекрасно', 'счастье', 'радость'];
    const negativeWords = ['плохо', 'ужасно', 'грусть', 'несчастье', 'зло'];

    let positiveCount = 0;
    let negativeCount = 0;

    positiveWords.forEach(word => {
        if (text.toLowerCase().includes(word)) {
            positiveCount++;
        }
    });

    negativeWords.forEach(word => {
        if (text.toLowerCase().includes(word)) {
            negativeCount++;
        }
    });

    if (positiveCount > negativeCount) {
        return 'Позитивный';
    } else if (negativeCount > positiveCount) {
        return 'Негативный';
    } else {
        return 'Нейтральный';
    }
}

function displayResult(sentiment) {
    const resultElement = document.getElementById('sentiment');
    resultElement.textContent = sentiment;

    const resultContainer = document.getElementById('result');
    resultContainer.style.display = 'block';

    // Изменение цвета в зависимости от результата
    if (sentiment === 'Позитивный') {
        resultElement.style.color = '#28a745';
    } else if (sentiment === 'Негативный') {
        resultElement.style.color = '#dc3545';
    } else {
        resultElement.style.color = '#6c757d';
    }
}