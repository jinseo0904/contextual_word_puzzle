// API URL - dynamically uses the current host and port
const API_URL = `${window.location.protocol}//${window.location.hostname}:${window.location.port}/api`;

// Game state
let gameState = {
    letters: [],
    centerLetter: '',
    validWords: [],
    pangrams: [],
    foundWords: new Set(),
    currentWord: '',
    outerLetters: [],
    currentScore: 0,
    maxScore: 0,
    selectedDay: ''
};

// Rank thresholds (percentages)
const RANKS = [
    { name: 'Beginner', threshold: 0 },
    { name: 'Good Start', threshold: 2 },
    { name: 'Moving Up', threshold: 5 },
    { name: 'Good', threshold: 8 },
    { name: 'Solid', threshold: 15 },
    { name: 'Nice', threshold: 25 },
    { name: 'Great', threshold: 40 },
    { name: 'Amazing', threshold: 50 },
    { name: 'Genius', threshold: 70 },
    { name: 'Queen Bee', threshold: 100 }
];

// DOM Elements
const setupPhase = document.getElementById('setup-phase');
const gamePhase = document.getElementById('game-phase');
const daySelector = document.getElementById('day-selector');
const seedInput = document.getElementById('seed-input');
const validateSeedBtn = document.getElementById('validate-seed-btn');
const seedResult = document.getElementById('seed-result');
const centerSelection = document.getElementById('center-selection');
const letterButtons = document.getElementById('letter-buttons');
const startGameBtn = document.getElementById('start-game-btn');
const wordInput = document.getElementById('word-input');
const currentWordDisplay = document.getElementById('current-word');
const submitBtn = document.getElementById('submit-btn');
const deleteBtn = document.getElementById('delete-btn');
const shuffleBtn = document.getElementById('shuffle-btn');
const wordMessage = document.getElementById('word-message');
const newGameBtn = document.getElementById('new-game-btn');
const toggleDebugBtn = document.getElementById('toggle-debug-btn');
const debugContent = document.getElementById('debug-content');
const generateHintsBtn = document.getElementById('generate-hints-btn');
const hintsLoading = document.getElementById('hints-loading');
const creativeHints = document.getElementById('creative-hints');

let selectedCenter = '';
let availableLetters = '';
let selectedDay = '';

// Setup Phase: Validate Seed
validateSeedBtn.addEventListener('click', async () => {
    const seed = seedInput.value.trim();
    
    if (!seed) {
        showMessage(seedResult, 'Please enter a seed word', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/validate-seed`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed })
        });
        
        const data = await response.json();
        
        if (data.valid) {
            availableLetters = data.letters;
            let message = `✓ Valid seed! Using ${data.letter_count} letters: ${data.letters.toUpperCase()}`;
            if (data.note) {
                message += `\n${data.note}`;
            }
            showMessage(seedResult, message, 'success');
            showCenterSelection(data.letters);
        } else {
            showMessage(seedResult, data.error, 'error');
        }
    } catch (error) {
        showMessage(seedResult, 'Error validating seed: ' + error.message, 'error');
    }
});

// Show center letter selection
function showCenterSelection(letters) {
    centerSelection.classList.remove('hidden');
    letterButtons.innerHTML = '';
    
    for (let letter of letters) {
        const btn = document.createElement('button');
        btn.className = 'letter-btn';
        btn.textContent = letter.toUpperCase();
        btn.onclick = () => selectCenter(letter, btn);
        letterButtons.appendChild(btn);
    }
}

// Select center letter
function selectCenter(letter, btn) {
    // Remove previous selection
    document.querySelectorAll('.letter-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selectedCenter = letter;
}

// Start game
startGameBtn.addEventListener('click', async () => {
    if (!selectedCenter) {
        alert('Please select a center letter');
        return;
    }
    
    const seed = seedInput.value.trim();
    selectedDay = daySelector.value;  // Capture selected day
    
    try {
        const response = await fetch(`${API_URL}/start-game`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                seed, 
                center: selectedCenter,
                day: selectedDay  // Pass selected day to backend
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            initializeGame(data);
            setupPhase.classList.remove('active');
            gamePhase.classList.add('active');
        } else {
            alert(data.error);
        }
    } catch (error) {
        alert('Error starting game: ' + error.message);
    }
});

// Initialize game
function initializeGame(data) {
    gameState.letters = data.letters.split('');
    gameState.centerLetter = data.center;
    gameState.validWords = data.valid_words;
    gameState.pangrams = data.pangrams;
    gameState.foundWords = new Set();
    gameState.currentWord = '';
    gameState.outerLetters = gameState.letters.filter(l => l !== gameState.centerLetter);
    gameState.currentScore = 0;
    gameState.maxScore = data.max_score;
    gameState.selectedDay = data.selected_day;
    
    // Update UI
    document.getElementById('game-letters').textContent = data.letters.toUpperCase();
    document.getElementById('game-center').textContent = data.center.toUpperCase();
    document.getElementById('total-count').textContent = data.total_words;
    document.getElementById('total-pangrams').textContent = data.pangrams.length;
    document.getElementById('found-count').textContent = '0';
    document.getElementById('pangram-count').textContent = '0';
    
    // Initialize score display
    document.getElementById('current-score').textContent = '0';
    document.getElementById('max-score').textContent = data.max_score;
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('rank-title').textContent = 'Beginner';
    
    // Create honeycomb
    createHoneycomb();
    
    // Setup debug panel
    setupDebugPanel();
    
    // Setup hints panel
    setupHintsPanel();
    
    // Clear found words
    document.getElementById('found-words-list').innerHTML = '';
    
    // Clear input
    wordInput.value = '';
    currentWordDisplay.textContent = '';
    wordMessage.textContent = '';
    
    // Update input placeholder with available letters
    wordInput.placeholder = `Use only these letters: ${data.letters.toUpperCase()}`;
    
    // Focus on input
    wordInput.focus();
}

// Setup hints panel with matrix
function setupHintsPanel() {
    // Update info
    document.getElementById('matrix-total').textContent = gameState.validWords.length;
    document.getElementById('matrix-points').textContent = gameState.maxScore;
    document.getElementById('matrix-pangrams').textContent = gameState.pangrams.length;
    
    // Build matrix
    buildMatrix();
    
    // Clear hints
    creativeHints.innerHTML = '<p style="color: #666; text-align: center;">Click "Generate Hints" to get AI-powered creative clues!</p>';
}

// Build the matrix grid
function buildMatrix() {
    // Build data structure
    const matrix = {};
    const twoLetterPairs = {};
    const lengths = new Set();
    
    gameState.validWords.forEach(wordData => {
        const word = wordData.word;
        const firstLetter = word[0].toUpperCase();
        const length = word.length;
        
        if (!matrix[firstLetter]) {
            matrix[firstLetter] = {};
        }
        matrix[firstLetter][length] = (matrix[firstLetter][length] || 0) + 1;
        lengths.add(length);
        
        // Two-letter pairs
        if (word.length >= 2) {
            const pair = word.substring(0, 2).toUpperCase();
            twoLetterPairs[pair] = (twoLetterPairs[pair] || 0) + 1;
        }
    });
    
    // Sort lengths
    const sortedLengths = Array.from(lengths).sort((a, b) => a - b);
    const letters = Object.keys(matrix).sort();
    
    // Find bingo (first letter with most words)
    let bingoLetter = '';
    let maxWords = 0;
    letters.forEach(letter => {
        const totalWords = Object.values(matrix[letter]).reduce((sum, count) => sum + count, 0);
        if (totalWords > maxWords) {
            maxWords = totalWords;
            bingoLetter = letter;
        }
    });
    document.getElementById('matrix-bingo').textContent = bingoLetter;
    
    // Build HTML table
    let html = '<table><thead><tr><th></th>';
    sortedLengths.forEach(len => {
        html += `<th>${len}</th>`;
    });
    html += '<th>Σ</th></tr></thead><tbody>';
    
    // Rows for each letter
    letters.forEach(letter => {
        html += `<tr><td class="row-header">${letter}:</td>`;
        let rowSum = 0;
        sortedLengths.forEach(len => {
            const count = matrix[letter][len] || 0;
            rowSum += count;
            const cellClass = count > 0 ? 'has-words' : '';
            html += `<td class="${cellClass}">${count > 0 ? count : '-'}</td>`;
        });
        html += `<td class="row-sum">${rowSum}</td></tr>`;
    });
    
    // Total row
    html += '<tr class="total-row"><td>Σ:</td>';
    sortedLengths.forEach(len => {
        let colSum = 0;
        letters.forEach(letter => {
            colSum += matrix[letter][len] || 0;
        });
        html += `<td>${colSum}</td>`;
    });
    html += `<td>${gameState.validWords.length}</td></tr>`;
    html += '</tbody></table>';
    
    document.getElementById('matrix-grid').innerHTML = html;
    
    // Two-letter list
    const twoLetterItems = Object.entries(twoLetterPairs)
        .sort((a, b) => a[0].localeCompare(b[0]))
        .map(([pair, count]) => `<span class="two-letter-item">${pair}-${count}</span>`)
        .join(' ');
    
    document.getElementById('two-letter-list').innerHTML = 
        `<strong>Two letter list:</strong><br>${twoLetterItems}`;
}

// Calculate points for a word
function calculatePoints(word, isPangram) {
    let points = word.length === 4 ? 1 : word.length;
    if (isPangram) {
        points += 7;
    }
    return points;
}

// Update score and rank
function updateScore(points) {
    // Get old rank
    const oldPercentage = (gameState.currentScore / gameState.maxScore) * 100;
    let oldRank = RANKS[0];
    for (let rank of RANKS) {
        if (oldPercentage >= rank.threshold) {
            oldRank = rank;
        }
    }
    
    // Update score
    gameState.currentScore += points;
    document.getElementById('current-score').textContent = gameState.currentScore;
    
    // Calculate new percentage
    const percentage = (gameState.currentScore / gameState.maxScore) * 100;
    
    // Update progress bar
    document.getElementById('progress-bar').style.width = `${percentage}%`;
    
    // Update rank
    let currentRank = RANKS[0];
    for (let rank of RANKS) {
        if (percentage >= rank.threshold) {
            currentRank = rank;
        }
    }
    
    const rankTitle = document.getElementById('rank-title');
    rankTitle.textContent = currentRank.name;
    
    // Animate if rank changed
    if (currentRank.name !== oldRank.name) {
        rankTitle.classList.add('rank-up');
        setTimeout(() => {
            rankTitle.classList.remove('rank-up');
        }, 500);
    }
}

// Create honeycomb hexagons
function createHoneycomb() {
    const svg = document.getElementById('honeycomb');
    svg.innerHTML = '';
    
    const hexSize = 45;
    const centerX = 150;
    const centerY = 150;
    
    // Calculate positions for 7 hexagons (1 center + 6 around)
    const positions = [
        { x: centerX, y: centerY, letter: gameState.centerLetter, isCenter: true }
    ];
    
    // Shuffle outer letters
    shuffleArray(gameState.outerLetters);
    
    // Calculate outer hexagon positions
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i - Math.PI / 2;
        const x = centerX + Math.cos(angle) * hexSize * 1.8;
        const y = centerY + Math.sin(angle) * hexSize * 1.8;
        positions.push({ 
            x, 
            y, 
            letter: gameState.outerLetters[i], 
            isCenter: false 
        });
    }
    
    // Draw hexagons
    positions.forEach(pos => {
        const hexagon = createHexagon(pos.x, pos.y, hexSize, pos.letter, pos.isCenter);
        svg.appendChild(hexagon);
    });
}

// Create a single hexagon
function createHexagon(x, y, size, letter, isCenter) {
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    group.classList.add('hex-cell');
    if (isCenter) group.classList.add('hex-center');
    else group.classList.add('hex-outer');
    
    // Create hexagon points
    const points = [];
    for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i;
        const px = x + size * Math.cos(angle);
        const py = y + size * Math.sin(angle);
        points.push(`${px},${py}`);
    }
    
    // Create polygon
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    polygon.setAttribute('points', points.join(' '));
    group.appendChild(polygon);
    
    // Create text
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.classList.add('hex-letter');
    text.setAttribute('x', x);
    text.setAttribute('y', y);
    text.textContent = letter.toUpperCase();
    group.appendChild(text);
    
    // Add click handler
    group.onclick = () => addLetter(letter);
    
    return group;
}

// Shuffle array
function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Add letter to current word
function addLetter(letter) {
    gameState.currentWord += letter;
    updateCurrentWordDisplay();
}

// Update current word display
function updateCurrentWordDisplay() {
    currentWordDisplay.textContent = gameState.currentWord.toUpperCase();
    wordInput.value = gameState.currentWord;
}

// Word input handling
wordInput.addEventListener('input', (e) => {
    const input = e.target.value.toLowerCase();
    // Filter out any letters not in the available letters
    const filtered = input.split('').filter(letter => 
        gameState.letters.includes(letter)
    ).join('');
    
    gameState.currentWord = filtered;
    updateCurrentWordDisplay();
});

wordInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        submitWord();
    }
});

// Delete button
deleteBtn.addEventListener('click', () => {
    gameState.currentWord = gameState.currentWord.slice(0, -1);
    updateCurrentWordDisplay();
});

// Shuffle button
shuffleBtn.addEventListener('click', () => {
    shuffleArray(gameState.outerLetters);
    createHoneycomb();
});

// Submit word
submitBtn.addEventListener('click', submitWord);

async function submitWord() {
    const word = gameState.currentWord.trim().toLowerCase();
    
    if (!word) {
        showWordMessage('Please enter a word', 'error');
        return;
    }
    
    // Check minimum length
    if (word.length < 4) {
        showWordMessage('Word must be at least 4 letters', 'error');
        return;
    }
    
    // Check if word only uses available letters
    const invalidLetters = word.split('').filter(letter => 
        !gameState.letters.includes(letter)
    );
    if (invalidLetters.length > 0) {
        showWordMessage('Word contains invalid letters', 'error');
        return;
    }
    
    // Check if word contains center letter
    if (!word.includes(gameState.centerLetter)) {
        showWordMessage('Word must contain center letter (' + gameState.centerLetter.toUpperCase() + ')', 'error');
        return;
    }
    
    if (gameState.foundWords.has(word)) {
        showWordMessage('Already found!', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/check-word`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                word, 
                valid_words: gameState.validWords,
                pangrams: gameState.pangrams,
                letters: gameState.letters.join(''),
                center: gameState.centerLetter
            })
        });
        
        const data = await response.json();
        
        if (data.valid) {
            gameState.foundWords.add(word);
            addFoundWord(word, data.is_pangram);
            showWordMessage(data.message, 'success');
            
            // Update score
            updateScore(data.points);
            
            // Update counts
            document.getElementById('found-count').textContent = gameState.foundWords.size;
            if (data.is_pangram) {
                const currentPangrams = parseInt(document.getElementById('pangram-count').textContent);
                document.getElementById('pangram-count').textContent = currentPangrams + 1;
            }
            
            // Check for Queen Bee
            if (gameState.foundWords.size === gameState.validWords.length) {
                setTimeout(() => {
                    showWordMessage('🎉 QUEEN BEE! You found all words! 🎉', 'success');
                }, 500);
            }
        } else {
            showWordMessage(data.message, 'error');
        }
        
        // Clear input
        gameState.currentWord = '';
        updateCurrentWordDisplay();
        wordInput.focus();
        
    } catch (error) {
        showWordMessage('Error checking word: ' + error.message, 'error');
    }
}

// Add found word to list
function addFoundWord(word, isPangram) {
    const foundWordsList = document.getElementById('found-words-list');
    const wordDiv = document.createElement('div');
    wordDiv.className = 'found-word' + (isPangram ? ' pangram' : '');
    wordDiv.textContent = word;
    foundWordsList.appendChild(wordDiv);
}

// Show word message
function showWordMessage(message, type) {
    wordMessage.textContent = message;
    wordMessage.className = `message ${type}`;
    setTimeout(() => {
        wordMessage.textContent = '';
        wordMessage.className = 'message';
    }, 3000);
}

// Show message helper
function showMessage(element, message, type) {
    element.textContent = message;
    element.className = `result-message ${type}`;
}

// Setup debug panel
function setupDebugPanel() {
    document.getElementById('debug-total').textContent = gameState.validWords.length;
    document.getElementById('debug-pangrams').textContent = gameState.pangrams.join(', ');
    
    const debugWordsList = document.getElementById('debug-words-list');
    debugWordsList.innerHTML = '';
    
    gameState.validWords.forEach(wordData => {
        const isPangram = gameState.pangrams.includes(wordData.word);
        const points = calculatePoints(wordData.word, isPangram);
        
        const wordDiv = document.createElement('div');
        wordDiv.className = 'debug-word';
        if (isPangram) {
            wordDiv.classList.add('pangram');
        }
        wordDiv.textContent = `${wordData.word} (${points}pts)`;
        wordDiv.title = `Points: ${points}\nFrequency: ${wordData.frequency.toExponential(2)}\n${wordData.definition.substring(0, 100)}...`;
        debugWordsList.appendChild(wordDiv);
    });
}

// Toggle debug panel
toggleDebugBtn.addEventListener('click', () => {
    debugContent.classList.toggle('hidden');
    toggleDebugBtn.textContent = debugContent.classList.contains('hidden') 
        ? 'Show All Words' 
        : 'Hide All Words';
});

// Generate hints button
generateHintsBtn.addEventListener('click', generateCreativeHints);

async function generateCreativeHints() {
    // Show loading
    hintsLoading.classList.remove('hidden');
    creativeHints.innerHTML = '';
    generateHintsBtn.disabled = true;
    generateHintsBtn.textContent = 'Generating...';
    
    try {
        const response = await fetch(`${API_URL}/generate-hints`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                words: gameState.validWords,
                day: gameState.selectedDay  // Pass selected day
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Show contextual mode indicator if applicable
            const contextualIndicator = document.getElementById('contextual-mode-indicator');
            if (data.contextual_mode && gameState.selectedDay) {
                document.getElementById('selected-day-display').textContent = `Day ${gameState.selectedDay}`;
                contextualIndicator.classList.remove('hidden');
            } else {
                contextualIndicator.classList.add('hidden');
            }
            
            displayCreativeHints(data.hints);
        } else {
            creativeHints.innerHTML = '<p style="color: red;">Error generating hints. Make sure Ollama is running with gpt-oss:20b model.</p>';
        }
    } catch (error) {
        creativeHints.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    } finally {
        hintsLoading.classList.add('hidden');
        generateHintsBtn.disabled = false;
        generateHintsBtn.textContent = 'Regenerate Hints';
    }
}

function displayCreativeHints(hintsByPrefix) {
    let html = '';
    
    const sortedPrefixes = Object.keys(hintsByPrefix).sort();
    
    sortedPrefixes.forEach(prefix => {
        const hints = hintsByPrefix[prefix];
        html += `<div class="hint-group">`;
        html += `<div class="hint-group-title">${prefix}</div>`;
        hints.forEach(hint => {
            html += `<div class="hint-item">${hint}</div>`;
        });
        html += `</div>`;
    });
    
    creativeHints.innerHTML = html;
}

// New game button
newGameBtn.addEventListener('click', () => {
    gamePhase.classList.remove('active');
    setupPhase.classList.add('active');
    seedInput.value = '';
    seedResult.textContent = '';
    seedResult.className = 'result-message';
    centerSelection.classList.add('hidden');
    selectedCenter = '';
    debugContent.classList.add('hidden');
    toggleDebugBtn.textContent = 'Show All Words';
});

