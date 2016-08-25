# synthes1.jl
# Translating to Julia script by Kibaek Kim, April 2016
#
# synthes1.mod   OOR2-AN-6-6
# AMPL coding: S. Leyffer, December 1998.
#
# Source: Test problem 1 (Synthesis of processing system) in 
# M. Duran & I.E. Grossmann,
# "An outer approximation algorithm for a class of mixed integer nonlinear
#  programs", Mathematical Programming 36, pp. 307-339, 1986.
#
# Number of variables:   6 (3 binary variables)  
# Number of constraints: 6
# Objective nonlinear
# Nonlinear constraints

push!(LOAD_PATH, "/nfs2/gkahvecioglu/Minotaur.jl/src")  # load path to Minotaur.jl
using JuMP
using Minotaur

m = Model(solver = MinotaurSolver());

x_U = [2,2,1];

@variable(m, x_U[i] >= x[i=1:3] >= 0);
@variable(m, y[1:3], Bin);

@NLobjective(m, Min, 
	5*y[1] + 6*y[2] + 8*y[3] + 10*x[1] - 7*x[3] - 18*log(x[2]+1)
	- 19.2*log(x[1]-x[2]+1) + 10);

@NLconstraint(m, 0.8*log(x[2] + 1) + 0.96*log(x[1] - x[2] + 1) - 0.8*x[3] >= 0);
@constraint(m, x[2] - x[1] <= 0);
@constraint(m, x[2] - 2*y[1] <= 0);
@constraint(m, x[1] - x[2] - 2*y[2] <= 0);
@NLconstraint(m, log(x[2] + 1) + 1.2*log(x[1] - x[2] + 1) - x[3] - 2*y[3] >= -2);
@constraint(m, y[1] + y[2] <= 1);

solve(m)
