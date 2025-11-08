// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <unordered_set>
#include <deque>
#include "isaaclab/manager/manager_term_cfg.h"
#include <iostream>

namespace isaaclab
{

using ObsMap = std::map<std::string, ObsFunc>;

inline ObsMap& observations_map() {
    static ObsMap instance;
    return instance;
}

#define REGISTER_OBSERVATION(name) \
    inline std::vector<float> name(ManagerBasedRLEnv* env); \
    inline struct name##_registrar { \
        name##_registrar() { observations_map()[#name] = name; } \
    } name##_registrar_instance; \
    inline std::vector<float> name(ManagerBasedRLEnv* env)


class ObservationManager
{
public:
    ObservationManager(YAML::Node cfg, ManagerBasedRLEnv* env)
    :cfg(cfg), env(env)
    {
        // Read global history length
        history_length = cfg["history_length"] ? cfg["history_length"].as<int>() : 1;
        _prapare_terms();
        _initialize_history_buffer();
    }

    void reset()
    {
        for(auto & term : obs_term_cfgs)
        {
            term.reset(term.func(this->env));
        }
        // Clear history buffer on reset
        obs_history_buffer.clear();
        obs_history_buffer.resize(history_length);
    }

    std::vector<float> compute()
    {
        // Step 1: Compute current observations (without history in individual terms)
        std::vector<float> current_obs;
        for(auto & term : obs_term_cfgs)
        {
            auto term_obs = term.func(this->env);  // Get raw observation
            
            // Apply scaling
            if (term.scale.size() > 0) {
                for (size_t i = 0; i < term_obs.size(); ++i) {
                    term_obs[i] *= term.scale[i];
                }
            }
            
            // Apply clipping
            if (term.clip.size() == 2) {
                for (auto& val : term_obs) {
                    val = std::max(term.clip[0], std::min(val, term.clip[1]));
                }
            }
            
            current_obs.insert(current_obs.end(), term_obs.begin(), term_obs.end());
        }
        
        // Step 2: Add current observation to history buffer
        obs_history_buffer.pop_front();
        obs_history_buffer.push_back(current_obs);
        
        // Step 3: Stack all historical observations
        std::vector<float> stacked_obs;
        for (const auto& obs : obs_history_buffer) {
            stacked_obs.insert(stacked_obs.end(), obs.begin(), obs.end());
        }
        
        return stacked_obs;
    }

    YAML::Node cfg;
    ManagerBasedRLEnv* env;
    int history_length;
    std::deque<std::vector<float>> obs_history_buffer;
    
private:
    void _initialize_history_buffer()
    {
        // Initialize history buffer with empty observations
        obs_history_buffer.resize(history_length);
    }

    void _prapare_terms()
    {
        for(auto it = this->cfg.begin(); it != this->cfg.end(); ++it)
        {
            // Skip non-term keys like "history_length"
            if (it->first.as<std::string>() == "history_length") {
                continue;
            }
            
            auto term_yaml_cfg = it->second;
            ObservationTermCfg term_cfg;
            // Don't set history_length for individual terms - we handle it globally
            term_cfg.history_length = 1;

            auto term_name = it->first.as<std::string>();
            if(observations_map()[term_name] == nullptr)
            {
                throw std::runtime_error("Observation term '" + term_name + "' is not registered.");
            }
            term_cfg.func = observations_map()[term_name];   

            auto obs = term_cfg.func(this->env);
            term_cfg.reset(obs);
            term_cfg.scale = term_yaml_cfg["scale"].as<std::vector<float>>();
            if(!term_yaml_cfg["clip"].IsNull()) {
                term_cfg.clip = term_yaml_cfg["clip"].as<std::vector<float>>();
            }

            this->obs_term_cfgs.push_back(term_cfg);
            std::cout << "Successfully loaded observation term: " << term_name << std::endl;
            
        }
    }
    
    std::vector<ObservationTermCfg> obs_term_cfgs;
};

};