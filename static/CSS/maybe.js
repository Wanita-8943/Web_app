const downloadButton = document.querySelector(`.download-button`)
const downloadIcon = document.querySelector(`.download-icon`)
const downloadLoader = document.querySelector(`.download-loader`)
const downloadCheckMark = document.querySelector(`.check-svg`)
const downloadText = document.querySelector(`.button-copy`)

downloadButton.addEventListener('click', () => {
    downloadIcon.classList.add(`hidden`)
    downloadLoader.classList.remove(`hidden`)
    downloadText.innerHTML = "DOWNLOADING";
}, { once: true })

downloadLoader.addEventListener('animationend', () => {
    downloadLoader.classList.add(`hidden`)
    downloadCheckMark.classList.remove(`hidden`)
    downloadText.innerHTML = "DOWNLOADED";
})