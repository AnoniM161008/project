document.getElementById('analyzeButton').addEventListener('click', function() {
    const text = document.getElementById('textInput').value;
    const sentiment = analyzeSentiment(text);
    displayResult(sentiment);
});

function analyzeSentiment(text) {
    // Пример простого анализа тональности
    const positiveWords = ["красивый", "стильный", "удобный", "практичный", "качественный", "надежный", "элегантный", "яркий", "уникальный", "модный", "функциональный", "прочный", "компактный", "эргономичный", "современный", "экологичный", "премиальный", "доступный", "инновационный", "универсальный", "эстетичный", "лаконичный", "роскошный", "гигиеничный", "лёгкий", "долговечный", "интуитивный", "атмосферный", "запоминающийся", "идеальный", "безупречно", "идеально", "бесперебойно", "бесшумно", "безукоризненно", "удобно", "плавно", "четко", "эффективно", "надежно", "ярко", "ровно", "аккуратно", "легко", "быстро", "приятно", "ровно", "гибко", "интуитивно", "эстетично", "супер"];
    const negativeWords = ["неудобный", "дешёвый", "некачественный", "хрупкий", "ненадёжный", "непрактичный", "некрасивый", "устаревший", "громоздкий", "неэргономичный", "нефункциональный", "выцветший", "скрипучий", "шаткий", "трещиноватый", "дефектный", "бракованный", "неприятный", "раздражающий", "неудобоваримый", "нелепый", "безвкусный", "топорный", "недолговечный", "неустойчивый", "несоответствующий", "перегруженный", "непродуманный", "неопрятный", "непривлекательный", "неудобно", "ненадежно", "хрупко", "громко", "медленно", "криво", "неприятно", "туго", "неравномерно", "слабо", "нечётко", "скрипуче", "шатко", "неаккуратно", "небрежно", "раздражающе", "непрактично", "нелепо", "неэстетично","ужасный"];

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