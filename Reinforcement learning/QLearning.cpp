#include<iostream>
#include<vector>
#include <cstdlib>
#include <ctime>
using namespace std;

/*
 * Machine Learning Profile:
 * ----------------------------------------------------------------------
 * - Model: Q-Learning (Model-Free, Value-Based Reinforcement Learning)
 * - Optimizer: Temporal Difference (TD) Target Value Iteration Update
 * - Policy Method: Epsilon-Greedy Exploration (epsilon-decay over steps)
 * - Environment Space: Deterministic 2D Grid World (8 X 6 discrete state space)
 * - Reward Structure: Step penalty (-1), Wall/blockade crash penalty (-6), Goal reward (+100)
 * - Convergence Guard: Cap at 150 maximum steps per training episode
 * ----------------------------------------------------------------------
 * This program implements a tabular Q-Learning agent from scratch to solve a 2D grid world pathfinding
 * problem with static obstacles. The environment maps state coordinates to a 3D tensor storing expected 
 * utility values for four discrete directional actions (Up, Down, Right, Left). During training over 1000 
 * episodes, the agent optimizes its policy using an epsilon-greedy approach that balances random actions 
 * with structural exploitation. By updating state-action matrix blocks via the Bellman Optimality equation, 
 * the model balances immediate step penalties against optimal future discount factors (gamma = 0.8). Upon 
 * training completion, the agent exploits its converged table tracking to derive the short-path trajectory.
 */

struct State {
    int row , col ;
    State(int row , int col ) : row(row), col(col){}
    bool operator==(const State& other) const {
        return (row == other.row && col == other.col);
    }
};
class QLearning{
    private:
        vector<vector<char>>grid;
        int GRIDROWS;      
        int GRIDCOLS; 
        State start;
        State end;  
        const int ACTIONS=4;
            double exploreRate = 1.0; 
            double minExploreRate = 0.03;
            double decay = 0.997;
        double alpha;
        double gamma;
        vector<vector<vector<double>>> Q;
        vector<vector<int>> rewards ;
        State move(State s , int action){
            State next=s;
            switch(action){
                case 0: if(s.row>0 && grid[s.row-1][s.col]!='#' )next.row--; break;
                case 1: if(s.row<GRIDROWS-1 && grid[s.row+1][s.col]!='#' )next.row++; break;
                case 2: if(s.col<GRIDCOLS-1 && grid[s.row][s.col+1]!='#' )next.col++; break;
                case 3: if(s.col>0 && grid[s.row][s.col-1]!='#' )next.col--; break;
            }
            return next;
        }
        int getRandomAction(){
            return rand()%ACTIONS;
        }
        void train(int episodes = 1){
            for ( int i=0 ;i < episodes; i++){
                State s = start;
                int countSteps=0;
                while ( true ){
                    countSteps++;
                    int action;
                    if((double)rand() / RAND_MAX < exploreRate){
                        action = getRandomAction();
                    }else{
                        action =0;
                        for( int a =1 ; a<ACTIONS ; a++){
                            if(Q[s.row][s.col][action]<Q[s.row][s.col][a]){
                                action =a ;
                            }
                        }

                    }
                    int hitWall=0;
                    State next = move(s,action);
                    (next==s)? hitWall=-5 : hitWall=0;
                    int reward = rewards[next.row][next.col] + hitWall;

                    double maxQ = Q[next.row][next.col][0];
                    for( int a = 1 ; a < ACTIONS ; ++a ){
                        if (Q[next.row][next.col][a] > maxQ)
                            maxQ = Q[next.row][next.col][a];
                    }
                    Q[s.row][s.col][action] += alpha * (reward + gamma * maxQ - Q[s.row][s.col][action]);

                    s = next;
                    exploreRate = max(minExploreRate, exploreRate * decay);

                    if (reward == 100 || countSteps == 150 ) break;
                }
            }
        }
    
    public:
        QLearning(vector<vector<char>> grid , State start , State end ,  double alpha ,double gamma):grid(grid) , start(start) , end(end)  , alpha( alpha) , gamma(gamma){
            GRIDROWS = grid.size();
            GRIDCOLS = grid[0].size();
            Q = vector<vector<vector<double>>>(GRIDROWS, vector<vector<double>>(GRIDCOLS, vector<double>(ACTIONS, 0)));
            rewards = vector<vector<int>>(GRIDROWS , vector<int> (GRIDCOLS ,-1));
            rewards[end.row][end.col] = 100;
        }
        void findePath(){
            State s=start;
            vector<vector<char>> pathgrid=grid;
            pathgrid[s.row][s.col] = 'S';
            while( s.row!=end.row || s.col!=end.col ){
                int bestaction=0;
                for( int i =1; i< ACTIONS ; i++){
                    if(Q[ s.row][s.col][bestaction] < Q[s.row][s.col][i]){
                        bestaction = i;
                    }
                }
                s=move(s,bestaction);
                pathgrid[s.row][s.col] = 'x';
            }
            pathgrid[s.row][s.col] = 'E';
            printGrid(pathgrid);
        }
        void printGrid(vector<vector<char>> grid){
            cout<<"\n";
            for (auto& row : grid){
                for(auto& col : row){
                    cout<< col<<"\t";
                }
                cout<<"\n";
            }
        }
        void run(){
            train(1000);
            findePath();
        }
};
int main (){
srand(time(nullptr));
system("cls");
std::cout<<"The shortes path is : \n";
vector<vector<char>> grid ={
    {'.','.','.','#','.','.'},
    {'.','.','#','#','.','.'},
    {'.','.','#','.','.','.'},
    {'.','.','#','.','#','.'},
    {'.','.','#','.','#','.'},
    {'.','.','.','.','#','.'},
    {'.','.','.','.','#','.'},
    {'.','.','.','.','#','.'},
};
State start(0,0);
State end(7,5);
double alpha=0.1;
double gamma=0.8;
QLearning qlearing(grid, start , end , alpha , gamma );

qlearing.run();

}