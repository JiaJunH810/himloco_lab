#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "isaaclab/devices/keyboard/keyboard.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize keyboard input
    FSMState::keyboard = std::make_shared<Keyboard>();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(new State_Passive(FSMMode::Passive));
    
    // Transition from Passive to FixStand on 'F' key
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                return FSMState::keyboard->key() == "f" && FSMState::keyboard->on_pressed;
            }, 
            (int)FSMMode::FixStand
        )
    );
    
    fsm->add(new State_FixStand(FSMMode::FixStand));
    
    // Transition from FixStand to RLBase on 'S' key
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                return FSMState::keyboard->key() == "s" && FSMState::keyboard->on_pressed;
            }, 
            FSMMode::Velocity
        )
    );
    
    // Transition from RLBase back to FixStand on 'Q' key
    fsm->add(new State_RLBase(FSMMode::Velocity, "Velocity"));
    fsm->states.back()->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ 
                return FSMState::keyboard->key() == "q" && FSMState::keyboard->on_pressed;
            }, 
            (int)FSMMode::FixStand
        )
    );

    std::cout << "\n=== Keyboard Control ===" << std::endl;
    std::cout << "  [F] - Enter FixStand mode" << std::endl;
    std::cout << "  [S] - Start RL control" << std::endl;
    std::cout << "  [Q] - Stop RL control (return to FixStand)" << std::endl;
    std::cout << "========================\n" << std::endl;

    while (true)
    {
        FSMState::keyboard->update();
        sleep(1);
    }
    
    return 0;
}

