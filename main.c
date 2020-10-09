/*
--------------------------------------------------
    91f7d09794d8da29f028e77df49d4907
    https://github.com/DaisyGAN/
--------------------------------------------------
    DaisyGANv3 / PortTalbot

    I changed from digesting 1228 messages every 333 seconds
    to; 42 messages every 9 seconds minimum.

    #1 :: I should have limited the max word len to 15 and
    pre-trim the newline char in a temp buffer, but since
    I have a wordlist of 16 chars max already I might as
    well waste the extra two bytes due to cache padding
    making temp buffer removal of the newline char superflous.
    Also 16 chars vs 15 chars per word seems to be make or break.
    Maybe I should just allow longer words in general ?
    But no because, I think words longer than 16 chars are
    most likely to be garble like url strings etc.
    This cache hit is not the end of the world.
*/

#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wformat-zero-length"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>

#define uint uint32_t

#define DIGEST_SIZE 16
#define FIRSTLAYER_SIZE 256
#define HIDDEN_SIZE 1024
#define DATA_SIZE 42
#define OUTPUT_QUOTES 3333
#define TABLE_SIZE_MAX 80000
#define MESSAGE_SIZE 256

#define TRAINING_LOOPS 1

const uint  _activator = 5;
const uint  _optimiser = 0;
const float _dropout = 0.5; //chance of neuron droput 0.3 = 30%
const float _lrate = 0.003;
const float _lmomentum = 0.9;
const float _wdecay = 0;

uint _log = 0;

struct
{
    float* data;
    float* momentum;
    float bias;
    float bias_momentum;
    uint weights;
}
typedef ptron;

// discriminator 
ptron d1[FIRSTLAYER_SIZE];
ptron d2[HIDDEN_SIZE];
ptron d3[HIDDEN_SIZE];
ptron d4;

// generator
ptron g1[FIRSTLAYER_SIZE];
ptron g2[HIDDEN_SIZE];
ptron g3[HIDDEN_SIZE];
ptron g4[DIGEST_SIZE];

// normalised training data
float digest[DATA_SIZE][DIGEST_SIZE] = {0};

//word lookup table / index
char wtable[TABLE_SIZE_MAX][DIGEST_SIZE+2] = {0}; //+2 for newline and null terminator ref[#1]
uint TABLE_SIZE = 0;
uint TABLE_SIZE_H = 0;


//*************************************
// utility functions
//*************************************

void loadTable(const char* file)
{
    FILE* f = fopen(file, "r");
    if(f)
    {
        uint index = 0;
        while(fgets(wtable[index], DIGEST_SIZE+2, f) != NULL) //+2
        {
            char* pos = strchr(wtable[index], '\n');
            if(pos != NULL)
                *pos = '\0';
            //printf("> %s : %u\n", wtable[index], index);
            index++;
            if(index == TABLE_SIZE_MAX)
                break;
        }
        TABLE_SIZE = index;
        TABLE_SIZE_H = TABLE_SIZE / 2;
        //printf("TABLE_SIZE: %u %u\n", TABLE_SIZE, TABLE_SIZE_H);
        fclose(f);
    }
}

float getWordNorm(const char* word)
{
    for(uint i = 0; i < TABLE_SIZE; i++)
    {
        if(strcmp(word, wtable[i]) == 0)
        {
            //printf("> %s : %u : %f\n", wtable[i], i, (((double)i) / (TABLE_SIZE_H))-1.0);
            return (((double)i) / (TABLE_SIZE_H))-1.0;
        }
    }
    return 0;
}

void saveWeights()
{
    FILE* f = fopen("weights.dat", "w");
    if(f != NULL)
    {
        for(uint i = 0; i < DIGEST_SIZE; i++)
        {
            if(fwrite(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        for(uint i = 0; i < HIDDEN_SIZE; i++)
        {
            if(fwrite(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");

            if(fwrite(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1w\n");
            
            if(fwrite(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
                printf("ERROR fwrite() in saveWeights() #1m\n");
        }

        if(fwrite(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        if(fwrite(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1w\n");
        
        if(fwrite(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
            printf("ERROR fwrite() in saveWeights() #1m\n");

        fclose(f);
    }
}

void loadWeights()
{
    FILE* f = fopen("weights.dat", "r");
    if(f == NULL)
    {
        printf("!!! no pre-existing weights where found, starting from random initialisation.\n\n\n-----------------\n");
        return;
    }

    for(uint i = 0; i < DIGEST_SIZE; i++)
    {
        while(fread(&d1[i].data[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].momentum[0], 1, d1[i].weights*sizeof(float), f) != d1[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d1[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d2[i].data[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].momentum[0], 1, d2[i].weights*sizeof(float), f) != d2[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d2[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    for(uint i = 0; i < HIDDEN_SIZE; i++)
    {
        while(fread(&d3[i].data[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].momentum[0], 1, d3[i].weights*sizeof(float), f) != d3[i].weights*sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

        while(fread(&d3[i].bias_momentum, 1, sizeof(float), f) != sizeof(float))
            sleep(333);
    }

    while(fread(&d4.data[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.momentum[0], 1, d4.weights*sizeof(float), f) != d4.weights*sizeof(float))
            sleep(333);

    while(fread(&d4.bias, 1, sizeof(float), f) != sizeof(float))
            sleep(333);

    while(fread(&d4.bias_momentum, 1, sizeof(float), f) != sizeof(float))
        sleep(333);

    fclose(f);
}

float qRandWeight(const float min, const float max)
{
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

float uRandWeight(const float min, const float max)
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    float pr = 0;
    while(pr == 0) //never return 0
    {
        const float rv = (float)rand();
        if(rv == 0)
            return min;
        const float rv2 = ( (rv / RAND_MAX) * (max-min) ) + min;
        pr = roundf(rv2 * 100) / 100; // two decimals of precision
    }
    return pr;
}

uint qRand(const uint min, const uint umax)
{
    static time_t ls = 0;
    if(time(0) > ls)
    {
        srand(time(0));
        ls = time(0) + 33;
    }
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}

uint uRand(const uint min, const uint umax)
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint s = 0;
    ssize_t result = read(f, &s, 4);
    srand(s);
    close(f);
    const int rv = rand();
    const uint max = umax + 1;
    if(rv == 0)
        return min;
    return ( ((float)rv / RAND_MAX) * (max-min) ) + min; //(rand()%(max-min))+min;
}

//https://stackoverflow.com/questions/30432856/best-way-to-get-number-of-lines-in-a-file-c
uint countLines(const char* file)
{
    uint lines = 0;
    FILE *fp = fopen(file, "r");
    if(fp != NULL)
    {
        while(EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp,"%*c")))
            ++lines;
        
        fclose(fp);
    }
    return lines;
}

void clearFile(const char* file)
{
    FILE *f = fopen(file, "w");
    if(f != NULL)
    {
        fprintf(f, "");
        fclose(f);
    }
}

void timestamp()
{
    const time_t ltime = time(0);
    printf("%s", asctime(localtime(&ltime)));
}


//*************************************
// create layer
//*************************************

void createPerceptron(ptron* p, const uint weights, const uint random)
{
    p->data = malloc(weights * sizeof(float));
    if(p->data == NULL)
    {
        printf("Perceptron creation failed (w)%u.\n", weights);
        return;
    }

    p->momentum = malloc(weights * sizeof(float));
    if(p->momentum == NULL)
    {
        printf("Perceptron creation failed (m)%u.\n", weights);
        return;
    }

    p->weights = weights;

    if(random == 1)
    {
        for(uint i = 0; i < weights; i++)
        {
            p->data[i] = qRandWeight(-1, 1);
            p->momentum[i] = 0;
        }
    }
    else if(random == 0)
    {
        memset(p->data, 0, weights * sizeof(float));
        memset(p->momentum, 0, weights * sizeof(float));
    }

    p->bias = qRandWeight(-1, 1);
    p->bias_momentum = 0;
}


//*************************************
// activation functions
// https://en.wikipedia.org/wiki/Activation_function
// https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/
// https://adl1995.github.io/an-overview-of-activation-functions-used-in-neural-networks.html
//*************************************

static inline float bipolarSigmoid(float x)
{
    return (1 - exp(-x)) / (1 + exp(-x));
}

static inline float fbiSigmoid(float x)
{
    return (1 - fabs(x)) / (1 + fabs(x));
}

static inline float arctan(float x)
{
    return atan(x);
}

static inline float lecun_tanh(float x)
{
    return 1.7159 * atan(0.666666667 * x);
}

static inline float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

static inline float fSigmoid(float x)
{
    return x / (1 + fabs(x));
    //return 0.5 * (x / (1 + abs(x))) + 0.5;
}

static inline float swish(float x)
{
    return x * sigmoid(x);
}

static inline float leakyReLU(float x)
{
    if(x < 0){x *= 0.01;}
    return x;
}

static inline float ReLU(float x)
{
    if(x < 0){x = 0;}
    return x;
}

static inline float ReLU6(float x)
{
    if(x < 0){x = 0;}
    if(x > 6){x = 6;}
    return x;
}

static inline float leakyReLU6(float x)
{
    if(x < 0){x *= 0.01;}
    if(x > 6){x = 6;}
    return x;
}

static inline float smoothReLU(float x) //aka softplus
{
    return log(1 + exp(x));
}

static inline float logit(float x)
{
    return log(x / (1 - x));
}

void softmax_transform(float* w, const uint32_t n)
{
    float d = 0;
    for(size_t i = 0; i < n; i++)
        d += exp(w[i]);

    for(size_t i = 0; i < n; i++)
        w[i] = exp(w[i]) / d;
}

float crossEntropy(const float predicted, const float expected) //log loss
{
    if(expected == 1)
      return -log(predicted);
    else
      return -log(1 - predicted);
}


//*************************************
// neural net / perceptron funcs
//*************************************

float doPerceptron(const float* in, ptron* p, const float error_override, const float eo)
{
//~~ Perceptron dropout [only during training]
    if(eo != -2 || error_override != 0)
    {
        if(_dropout != 0)
        {
            if(uRandWeight(0.01, 1) <= _dropout)
                return 0;
        }
    }

//~~ Query perceptron
    //Sum inputs mutliplied by weights
    float ro = 0;
    for(uint i = 0; i < p->weights; i++)
        ro += in[i] * p->data[i];
    ro += p->bias;

//~~ Activation Function
    if(_activator == 1)
        ro = sigmoid(ro);
    else if(_activator == 2)
    {
        if(ro < 0){ro = 0;}
    }
    else if(_activator == 3)
        ro = leakyReLU(ro);
    else if(_activator == 4)
        ro = smoothReLU(ro);
    else if(_activator == 5)
        ro = arctan(ro);
    else if(_activator == 6)
        ro = tanh(ro);
    else if(_activator == 7)
        ro = lecun_tanh(ro);
    else if(_activator == 8)
        ro = logit(ro);
    else if(_activator == 9)
        ro = bipolarSigmoid(ro);
    else if(_activator == 10)
        ro = fbiSigmoid(ro);
    else if(_activator == 11)
        ro = fSigmoid(ro);
    else if(_activator == 12)
        ro = swish(ro);
    else if(_activator == 13)
        ro = leakyReLU6(ro);
    else if(_activator == 14)
        ro = ReLU6(ro);

//~~ Teach perceptron
    if(eo != -2 || error_override != 0)
    {
        const float error = error_override == 0 ? eo - ro : error_override - ro; // error gradient
        for(uint i = 0; i < p->weights; i++)
        {
            // Weight Decay
            if(_wdecay != 0)
                p->data[i] *= (1.0 - _wdecay);

            // Regular Gradient Descent
            if(_optimiser == 0)
                p->data[i] += error * in[i] * _lrate;

            // Regular Momentum
            if(_optimiser == 1)
            {
                p->momentum[i] = _lmomentum * p->momentum[i] + (error * in[i] * _lrate);
                p->data[i] += p->momentum[i];
            }
            
            // Nesterov (NAG) momentum
            if(_optimiser == 2)
            {
                const float g = error * in[i] * _lrate;
                const float l = p->data[i] + p->momentum[i];
                p->momentum[i] = _lmomentum * p->momentum[i] + g * l;
                p->data[i] += p->momentum[i];
            }
        }

        // Weight Decay
        if(_wdecay != 0)
            p->bias *= (1.0 - _wdecay);

        // Regular Gradient Descent
        if(_optimiser == 0)
            p->bias += error * _lrate;

        // Regular Momentum
        if(_optimiser == 1)
        {
            p->bias_momentum = _lmomentum * p->bias_momentum + (error * _lrate);
            p->bias += p->bias_momentum;
        }
        
        // Nesterov (NAG) momentum
        if(_optimiser == 2)
        {
            p->bias_momentum = _lmomentum * p->bias_momentum + (error * _lrate) * (p->bias + p->bias_momentum);
            p->bias += p->bias_momentum;
        }
    }

//~~ Return output
    return ro;
}


//*************************************
// network training functions
//*************************************

float doDiscriminator(const float* input, const float eo)
{
    // layer one, inputs (fc)
    float o1[FIRSTLAYER_SIZE];
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1[i] = doPerceptron(input, &d1[i], 0, eo);

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = doPerceptron(&o1[0], &d2[i], 0, eo);

    // layer three, hidden (fc)
    float o3[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3[i] = doPerceptron(&o2[0], &d3[i], 0, eo);
    
    // layer four, output (fc compression)
    return doPerceptron(&o3[0], &d4, 0, eo);
}

void doGenerator(const float error, const float* input, float* output)
{
    // layer one, inputs (fc)
    float o1[FIRSTLAYER_SIZE];
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        o1[i] = doPerceptron(input, &g1[i], error, -2);

    // layer two, hidden (fc expansion)
    float o2[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o2[i] = doPerceptron(&o1[0], &g2[i], error, -2);

    // layer three, hidden (fc)
    float o3[HIDDEN_SIZE];
    for(int i = 0; i < HIDDEN_SIZE; i++)
        o3[i] = doPerceptron(&o2[0], &g3[i], error, -2);
    
    // layer four, output (fc compression)
    for(int i = 0; i < DIGEST_SIZE; i++)
        output[i] = doPerceptron(&o3[0], &g4[i], error, -2);
}

/*
--------------------------------------------------
    Defunct Backprop Method Start
--------------------------------------------------
*/
// void doSGD(float* weight, float* momentum, const float error, const float _lrate)
// {
//     // Regular Gradient Descent
//     if(_optimiser == 0)
//     {
//         weight[0] += error * _lrate;
//     }

//     // Regular Momentum
//     if(_optimiser == 1)
//     {
//         momentum[0] = _lmomentum * momentum[0] + (error * _lrate);
//         weight[0] += *momentum;
//     }
    
//     // Nesterov (NAG) momentum
//     if(_optimiser == 2)
//     {
//         momentum[0] = _lmomentum * momentum[0] + (error * _lrate) * (weight[0] + momentum[0]);
//         weight[0] += momentum[0];
//     }
// }

// void backpropGenerator(const float error, const float lrate)
// {
//     // layer one, inputs (fc)
//     for(int i = 0; i < DIGEST_SIZE; i++)
//     {
//         const float layer_error = (1-(1/i))*error;

//         for(int j = 0; j < g1[i].weights; j++)
//             doSGD(&g1[i].data[j], &g1[i].momentum[j], layer_error, lrate);
        
//         doSGD(&g1[i].bias, &g1[i].bias_momentum, layer_error, lrate);
//     }

//     // layer two, hidden (fc expansion)
//     for(int i = 0; i < HIDDEN_SIZE; i++)
//     {
//         const float layer_error = (1-(1/i))*error;

//         for(int j = 0; j < g2[i].weights; j++)
//             doSGD(&g2[i].data[j], &g2[i].momentum[j], layer_error, lrate);
        
//         doSGD(&g2[i].bias, &g2[i].bias_momentum, layer_error, lrate);
//     }
    
//     // layer three, output (fc compression)
//     for(int i = 0; i < DIGEST_SIZE; i++)
//     {
//         const float layer_error = (1-(1/i))*error;

//         for(int j = 0; j < g3[i].weights; j++)
//             doSGD(&g3[i].data[j], &g3[i].momentum[j], layer_error, lrate);
        
//         doSGD(&g3[i].bias, &g3[i].bias_momentum, layer_error, lrate);
//     }
// }
/*
--------------------------------------------------
    Defunct Backprop Method End
--------------------------------------------------
*/

float rmseDiscriminator()
{
    float squaremean = 0;
    for(int i = 0; i < DATA_SIZE; i++)
    {
        const float r = 1 - doDiscriminator(&digest[i][0], -2);
        squaremean += r*r;
    }
    squaremean /= DATA_SIZE;
    return sqrt(squaremean);
}

void trainDataset(const char* file)
{
    // read training data [every input is truncated to 256 characters]
    FILE* f = fopen(file, "r");
    if(f)
    {
        char line[MESSAGE_SIZE];
        uint index = 0;
        while(fgets(line, MESSAGE_SIZE, f) != NULL)
        {
            char* pos = strchr(line, '\n');
            if(pos != NULL)
                *pos = '\0';
            uint i = 0;
            char* w = strtok(line, " ");
            //printf("> %s : %i\n", line, index);
            while(w != NULL)
            {
                digest[index][i] = getWordNorm(w); //normalise
                //printf("> %s : %f\n", w, digest[index][i]);
                w = strtok(NULL, " ");
                i++;
            }

            index++;
            if(index == DATA_SIZE)
                break;
        }
        fclose(f);
    }

    printf("Training Data Loaded\n");

    // train discriminator
    for(int j = 0; j < TRAINING_LOOPS; j++)
    {
        for(int i = 0; i < DATA_SIZE; i++)
        {
            // train discriminator on data
            doDiscriminator(&digest[i][0], 1.57079632679);

            // detrain discriminator on random word sequences 
            float output[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE-1);
            for(int i = 0; i < len; i++)
                output[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0; //uRandWeight(-1, 1);
            doDiscriminator(&output[0], 0);

             if(_log == 1)
                printf("Training Iteration (%u / %u) [%u / %u]\n RAND | REAL\n", i+1, DATA_SIZE, j+1, TRAINING_LOOPS);

            if(_log == 1)
            {
                for(int k = 0; k < DIGEST_SIZE; k++)
                    printf("%+.2f : %+.2f\n", output[k], digest[i][k]);

                printf("\n");
            }

            //usleep(100000);
        }

        printf("RMSE: %f\n", rmseDiscriminator());
    }

    // save weights
    saveWeights();
    //printf("Weights Saved: weights.dat\n");
}

void consoleAsk()
{
    // what percentage human is this ?
    while(1)
    {
        char str[MESSAGE_SIZE] = {0};
        float nstr[DIGEST_SIZE] = {0};
        printf(": ");
        fgets(str, MESSAGE_SIZE, stdin);
        str[strlen(str)-1] = 0x00; //remove '\n'

        //normalise words
        uint i = 0;
        char* w = strtok(str, " ");
        while(w != NULL)
        {
            nstr[i] = getWordNorm(w);
            w = strtok(NULL, " ");
            i++;
        }

        const float r = doDiscriminator(nstr, -2);
        printf("This is %.2f%% (%.2f) Human.\n", (r+1.57079632679)*31.830988618, r); //arctan conversion
    }
}

float isHuman(char* str)
{
    float nstr[DIGEST_SIZE] = {0};

    //normalise words
    uint i = 0;
    char* w = strtok(str, " ");
    while(w != NULL)
    {
        nstr[i] = getWordNorm(w);
        printf("> %s : %f\n", w, nstr[i]);
        w = strtok(NULL, " ");
        i++;
    }

    const float r = doDiscriminator(nstr, -2);
    return (r+1.57079632679)*31.830988618; //arctan conversion
}

float rndScentence()
{
    float nstr[DIGEST_SIZE] = {0};
    const int len = uRand(1, DIGEST_SIZE-1);
    for(int i = 0; i < len; i++)
        nstr[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

    for(int i = 0; i < DIGEST_SIZE; i++)
    {
        const uint ind = (nstr[i]+1.0)*TABLE_SIZE_H;
        if(nstr[i] != 0)
            printf("%s (%.2f) ", wtable[ind], nstr[i]);
    }

    printf("\n");

    const float r = doDiscriminator(nstr, -2);
    return (r+1.57079632679)*31.830988618; //arctan conversion
}

void trainGenerator(const char* file)
{
    // train generator
    uint index = 0;
    float last_error = 0;
    FILE* f = fopen(file, "w");
    if(f != NULL)
    {
        for(int k = 0; k < OUTPUT_QUOTES; k++)
        {
            // random generator input
            float input[DIGEST_SIZE] = {0};
            const int len = uRand(1, DIGEST_SIZE-1);
            for(int i = 0; i < len; i++)
                input[i] = (((double)uRand(0, TABLE_SIZE))/TABLE_SIZE_H)-1.0;

            // do generator
            float output[DIGEST_SIZE] = {0};
            doGenerator(last_error * 0.3, &input[0], &output[0]);

            // feed generator output into discriminator input, take the error, sigmoid it to 0-1, take the loss, put it back through as the error for the next generation
            last_error = crossEntropy(sigmoid(3.141592654 - (doDiscriminator(&output[0], -2) + 1.57079632679)), 1);

            // convert output to string of words
            if(_log == 1)
                printf("[%.2f] ", last_error);
            const double pre1 = TABLE_SIZE / 3.141592654;
            int last_index = -1;
            for(int i = 0; i < DIGEST_SIZE; i++)
            {
                const double ind = (( ((double)output[i])+1.57079632679 ) *pre1)+0.5; //arctan conversion
                if(output[i] != 0.0 && ind < TABLE_SIZE && ind > 0)
                {
                    const int new_index = (int)ind;
                    if(new_index == last_index) //stop the bot repeating words
                        continue;
                    last_index = new_index;
                    fprintf(f, "%s ", wtable[new_index]);
                    if(_log == 1)
                        printf("%s ", wtable[new_index]); //printf("%s (%i) ", wtable[(int)ind], (int)ind);
                }
            }
            fprintf(f, "\n");
            if(_log == 1)
                printf("\n");

            // back prop the generator [defunct method of backprop]
            //backpropGenerator(last_error, 0.001);

            // output error
            //printf("[%u]ERROR: %.2f\n\n", index, last_error);
            //index++;
            //sleep(1);
        }
    }
}


//*************************************
// program entry point
//*************************************

int main(int argc, char *argv[])
{
    // init discriminator
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&d1[i], DIGEST_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d2[i], FIRSTLAYER_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&d3[i], HIDDEN_SIZE, 1);
    createPerceptron(&d4, HIDDEN_SIZE, 1);

    // init generator
    for(int i = 0; i < FIRSTLAYER_SIZE; i++)
        createPerceptron(&g1[i], DIGEST_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&g2[i], FIRSTLAYER_SIZE, 1);
    for(int i = 0; i < HIDDEN_SIZE; i++)
        createPerceptron(&g3[i], HIDDEN_SIZE, 1);
    for(int i = 0; i < DIGEST_SIZE; i++)
        createPerceptron(&g4[i], HIDDEN_SIZE, 1);

    // load lookup table
    loadTable("tgdict.txt");

    // are we issuing any commands?
    if(argc == 3)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            trainDataset(argv[2]);
            exit(0);
        }
    }

    if(argc == 2)
    {
        if(strcmp(argv[1], "retrain") == 0)
        {
            _log = 1;
            trainDataset("tgmsg.txt");
            exit(0);
        }

        loadWeights();

        if(strcmp(argv[1], "ask") == 0)
            consoleAsk();

        if(strcmp(argv[1], "rnd") == 0)
        {
            printf("> %.2f\n", rndScentence());
            exit(0);
        }

        if(strcmp(argv[1], "gen") == 0)
        {
            _log = 1;
            trainGenerator("out.txt");
            exit(0);
        }

        if(strcmp(argv[1], "rndloop") == 0)
        {
            while(1)
            {
                printf("> %.2f\n\n", rndScentence());
                //usleep(100000);
            }
        }

        char in[MESSAGE_SIZE] = {0};
        snprintf(in, MESSAGE_SIZE, "%s", argv[1]);
        printf("%.2f\n", isHuman(in));
        exit(0);
    }

    // no commands ? then I would suppose we are running the generator
    loadWeights();

    // main loop
    printf("Running ! ...\n\n");
    while(1)
    {
        if(countLines("tgmsg.txt") >= DATA_SIZE)
        {
            timestamp();
            const time_t st = time(0);
            loadTable("tgdict.txt");
            trainDataset("tgmsg.txt");
            clearFile("tgmsg.txt");
            trainGenerator("out.txt");
            printf("Just generated a new dataset.\n");
            timestamp();
            printf("Time Taken: %.2f mins\n\n", ((double)(time(0)-st)) / 60.0);
        }

        sleep(9);
    }

    // done
    return 0;
}

