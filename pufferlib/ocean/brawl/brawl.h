#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

/*
PLAN FOR ENV

Start with 2 agents, 3 weapons:
-Shotgun (very high damage, close range)
-SMG (medium damage, medium range)
-Sniper (high damage, long range, slow attack speed)

Agents fight each other, goal is to train a strong fighter.

Goals for v1:
2 agents, one gun (SMG), simple map (2 walls)

The gun will require reloading (40 ammo).

Agents have 100 health (will all be normalized later)
Each shot w/ SMG does 10 damage
Can fire 5 rounds / sec
Reloading takes 1 second
Need to define movement speed, should take around 15 seconds to traverse map

What do agents need to observe?
-Their own position x and y (norm)
-Their ammo (norm)
-Their health (norm)
-Their heading (norm)
-Enemy position x and y (normalized)
-Enemy health (norm)
-Enemy heading (norm)

Use raycasting for walls. They'll be told if each ray hits a wall or a player. (probably need LSTM bc hidden info)
Raycast 16 rays in 360 degrees around agent

What actions can the agents take?
-Move left, right, up, down (discrete)
-Fire (discrete)
-Change heading (continuous)

Environment parameters:
-Ammo amount
-Reload time
-Agent health
-Agent move speed

What I need to learn:
How to do raycasting (rotate rays based on agent heading)
How to set it up so I can play vs an agent
How to render 2 agents at a time
How to do agent vs agent training, checkpointing, avoid cycling etc
How to set up and train LSTM in pufferlib (try MLP then LSTM?)

V1.0:
Two agents, raycasting for observations, no walls, only SMG.

V1.1:
-Add walls

V1.2:
-Add other guns

V1.3:
-Add more agents

V1.4:
-Add more maps
*/

#define PI 3.14159265358

const size_t env_steps = 10000;
const float agent_health = 100;
const int SMG_damage = 10;
const float SMG_amo = 40;
const float SMG_reload_time = 2000; // ms
const float move_speed = 0.002; // % of map per frame

int NOOP = 0;
int LEFT = 1;
int RIGHT = 2;
int SHOOT = 3;
int RELOAD = 4;

typedef struct {
    float perf;
    float score;
    float episode_return;
    float episode_length;

    float n;
} Log;

typedef struct {
    float x;
    float y;
    float heading;
    float health;
    int SMG_amo;
    float radius; // agents are circles
} Agent;

typedef struct {
    Log log;
    Agent* agents; // only 2
    Log* agent_logs; // only 2
    float* observations;
    int* actions; // 2 actions per agent (one for heading)
    float* rewards;
    unsigned char* terminals;
    int width;
    int height;
    int num_agents; // 2 for V1
    float reward_attack;
    float reward_damage;
    float reward_death;
    float reward_kill;
} Brawl;

void add_log(Brawl* env, int agent_id) {
    env->log.perf += env->agent_logs[agent_id].perf;
    env->log.score += env->agent_logs[agent_id].score;
    env->log.episode_length += env->agent_logs[agent_id].episode_length;
    env->log.episode_return += env->agent_logs[agent_id].episode_return;
    env->log.n++;
}

void init(Brawl* env) {
    env->agents = (Agent*)calloc(env->num_agents, sizeof(Agent));
    env->agent_logs = (Log*)calloc(env->num_agents, sizeof(Log));
}

void c_close(Brawl* env) {
    free(env->agents);
    free(env->agent_logs);
}

void free_brawl(Brawl* env) { // do we not free terminals?
    c_close(env);
    free(env->observations);
    free(env->actions);
    free(env->rewards);
}

float max(float x,  float y) {
    if(x > y) {
        return x;
    }
    return y;
}

float min(float x, float y) {
    if(x < y) {
        return x;
    }
    return y;
}

// casts ray from (x, y) in direction heading (expects heading in 0 to 2pi)
static float ray_aabb_first_hit(Brawl* env, float x, float y, float cx, float cy,
    float heading, float xmin, float ymin, float xmax, float ymax, float eps) {
    float dx = cosf(heading);
    float dy = sinf(heading);
    
    float tmin = -1;
    float tmax = 1;

    float ocx = x - cx;
    float ocy = y - cy;

    if(abs(dx) < eps) {
        if(x < xmin || x > xmax) {
            return -2; // figure out better error code
        }
    } else {
        float tx1 = (xmin - x) / dx;
        float tx2 = (xmax - x) / dx;
        float tmin = max(tmin, min(tx1, tx2));
        float tmax = min(tmax, max(tx1, tx2));
    }

    if(abs(dy) < eps) {
        if(y < ymin || y > ymax) {
            return -2; // figure out better error code
        }
    } else {
        float ty1 = (ymin - y) / dy;
        float ty2 = (ymax - y) / dy;
        float tmin = max(tmin, min(ty1, ty2));
        float tmax = min(tmax, max(ty1, ty2));
    }

    if(tmax > tmin) {
        return -2;
    }

    if(tmax < 0) {
        return -2;
    }

    return (tmin >= 0) ? tmin : tmax;
}

static float ray_circle_first_hit(Brawl* env, float x, float y, float cx, float cy, float heading, 
    float xmin, float ymin, float xmax, float ymax, float r, float eps) {
        float dx = cosf(heading);
        float dy = sinf(heading);
        
        float ocx = x - cx;
        float ocy = y - cy;
}

void compute_observations(Brawl* env) {
    /*
    Need to cast rays with start x, y and heading
    */

}

