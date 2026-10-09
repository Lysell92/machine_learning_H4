let mutationRate = 0.0093;
let populationSize = 1150;
let population = [];
let maxGeneration = 700;
let fitnessEvaluations = 0;
let target = "to be or not to be"

function setup() {
    for (let i = 0; i < populationSize; i++) {
        population[i] = new DNA(target.length)
    }
}

function draw() {
    
    for (let phrase of population) {
        phrase.calculateFitness(target);
        fitnessEvaluations++;
    }

    let best = population[0];

    for (let phrase of population){
        if (phrase.fitness > best.fitness) {
            best = phrase;
        }
    }
    
    console.log(best.genes.join(""), best.fitness);

    if (best.fitness === 1) {    
        return true;            
    }

    // Preserve the best individual.
    let matingPool = [];

    for (let phrase of population) {
        let n = Math.max(1, Math.floor(phrase.fitness * 100));
    
        for (let j = 0; j < n; j++) {
            matingPool.push(phrase);
        }
    }

    if (matingPool.length === 0) {
        console.log("Mating pool is empty!");
        return false;
    }

    let nextPopulation = []

    nextPopulation.push(best);

    while (nextPopulation.length < population.length){
        let parentA = randomFrom(matingPool);
        let parentB = randomFrom(matingPool);
        
        let child = parentA.crossover(parentB);
        child.mutate(mutationRate);

        nextPopulation.push(child);
    }

    population = nextPopulation;

    return false;
}

class DNA {
    constructor(length) {

    this.genes = [];

    for (let i = 0; i < length; i++) {
        this.genes[i] = randomCharacter(); 
    }
    
    this.fitness = 0;
  }

    calculateFitness(target){
        let score = 0;
        
        for(let i = 0; i < this.genes.length; i++) {
            if (this.genes[i] === target.charAt(i)) {
                score++;
            }
        }
        this.fitness = score / target.length
    }
    crossover(partner) {
        let child = new DNA(this.genes.length);
        
        let midpoint = Math.floor(Math.random() * this.genes.length)
        for (let i = 0; i < this.genes.length; i++) {
            if (i < midpoint) {
                child.genes[i] = this.genes[i];
            }
            else {
                child.genes[i] = partner.genes[i];
            }
        }
        return child;
    }

    mutate(mutationRate) {
        for (let i = 0; i < this.genes.length; i++) {
            if (Math.random() < mutationRate) {
                this.genes[i] = randomCharacter();
            }
        }
    }
}


function randomCharacter() {
    let c = Math.floor(Math.random() * (127 - 32) + 32);
    return String.fromCharCode(c);
}
  
function randomFrom(array) {
    let index = Math.floor(Math.random() * array.length);
    return array[index];
}


setup();

let startTime = performance.now();

for (let generation = 0; generation < maxGeneration; generation ++) {
    let finished = draw();

    if (finished) {

        let elapsed = performance.now() - startTime;
        
        console.log(`Target reached in generation: ${generation}`);
        console.log(`Fitness evaluations: ${fitnessEvaluations}`);
        console.log(`Time: ${elapsed.toFixed(2)} ms`);
        
        break;
    }
}