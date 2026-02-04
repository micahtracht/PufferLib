#include <stdlib.h>
#include <string.h>
#include "raylib.h"

const unsigned char NOOP = 0;
const unsigned char UP = 1;
const unsigned char DOWN = 2;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char WALL = 2;
const unsigned char DANGER = 3; // agent and wall overlap

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;

    float n; // must be last field
} Log;

//use unsigned char to save memory (think of as short)
typedef struct {
    Log log;
    unsigned char* observations;
    int* actions;
    float* rewards;
    unsigned char* terminals;
    int tick;
    int h;
    float episode_return;

} flappy;

void add_log(flappy* env) {
    env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
    env->log.score += env->rewards[0];
    env->log.episode_length += env->tick;
    env->log.episode_return += env->episode_return;
    env->log.n++;
}

//required func
void c_reset(flappy* env) {
    env->episode_return = 0;
    size_t num_tiles = 2;
    memset(env->observations, 0, num_tiles * sizeof(unsigned char)); // I can just use calloc, right?
    env->h = 0;
    env->tick = 0;
    int target_idx = rand() % 2;
    env->observations[0] = AGENT;
    if(target_idx == 0) {
        env->observations[0] = DANGER;
    } else {
        env->observations[target_idx] = WALL;
    }
    env->episode_return = 0;
}

//required
void c_step(flappy* env) {
    env->tick++;
    
    int atn = env->actions[0];
    env->terminals[0] = 0;
    env->rewards[0] = 0;

    if(atn==DOWN) {
        env->h++;
    }
    if(atn==UP) {
        env->h--;
    }

    if(env->h < 0 || env->h > 1
    || env->observations[env->h] == WALL // may change to a 2D layout to accommodate multiple agents
    || env->observations[env->h] == DANGER) {
        env->terminals[0] = 1;
        env->rewards[0] = -1.0f;
        env->episode_return += -1.0f;
        add_log(env);
        c_reset(env);
        return;
    }
    else if(env->tick > 100) {
        env->terminals[0] = 1;
        env->rewards[0] = 1.0f;
        env->episode_return += 1.0f;
        add_log(env);
        c_reset(env);
        return;
    } else {
        env->observations[0] = EMPTY;
        env->observations[1] = EMPTY;
        env->observations[env->h] = AGENT;
        env->rewards[0] = 0.15f;
        env->episode_return += 0.15f;
    }
    unsigned char random = rand() % 2;
    if(env->observations[random] == AGENT) {
        env->observations[random] = DANGER;
    } else {
        env->observations[random] = WALL;
    }
}

void c_render(flappy* env) {
    return; // not implementing this yet
}

// DO NOT FREE
void c_close(flappy* env) {
    if(IsWindowReady()) {
        CloseWindow();
    }
}