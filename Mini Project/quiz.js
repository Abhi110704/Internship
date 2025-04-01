const quizData = [{
    question: "Initially named High Jump, this movie is inspired by the biography of Akira Kurosawa – the famous Japanese filmmaker. What is it?",
    a: "Luck By Chance",
    b: "3 Idiots",
    c: "Dangal",
    d: "Taare Zameen Par",
    correct: "d",
},
{
    question: "What movie genre does “Hungama” belong to?",
    a: "Comedy",
    b: "Horror",
    c: "Romance",
    d: "Mystery",
    correct: "a",
},
{
    question: "Who created the original story for Shehenshah, one of the most iconic movies in Bollywood history?",
    a: "Dilip Kumar",
    b: "Jaya Bacchan",
    c: "Shashi Kapoor",
    d: "none of the above",
    correct: "b",
},
{
    question: "Who is the director of “Devdas”, “Aishwarya Rai”, and Shahrukh Khan?",
    a: "Karan Johar",
    b: "Sanjay Leela Bhansali",
    c: "Aashutosh Gowariker",
    d: "Mahesh Bhatt",
    correct: "b",
}
];
let index = 0;
let correct = 0,
incorrect = 0,
total = quizData.length;
let questionBox = document.getElementById("questionBox");
let allInputs = document.querySelectorAll("input[type='radio']")
const loadQuestion = () => {
if (total === index) {
    return quizEnd()
}
reset()
const data = quizData[index]
questionBox.innerHTML = `${index + 1}) ${data.question}`
allInputs[0].nextElementSibling.innerText = data.a
allInputs[1].nextElementSibling.innerText = data.b
allInputs[2].nextElementSibling.innerText = data.c
allInputs[3].nextElementSibling.innerText = data.d
}

document.querySelector("#submit").addEventListener(
"click",
function() {
    const data = quizData[index]
    const ans = getAnswer()
    if (ans === data.correct) {
        correct++;
    } else {
        incorrect++;
    }
    index++;
    loadQuestion()
}
)

const getAnswer = () => {
let ans;
allInputs.forEach(
    (inputEl) => {
        if (inputEl.checked) {
            ans = inputEl.value;
        }
    }
)
return ans;
}

const reset = () => {
allInputs.forEach(
    (inputEl) => {
        inputEl.checked = false;
    }
)
}

const quizEnd = () => {
document.getElementsByClassName("container")[0].innerHTML = `
    <div class="col">
        <h3 class="w-100"> Hii, you've scored ${correct} / ${total} </h3>
    </div>
`
}
loadQuestion(index);
