// virus example
// gxn/dxp 29/06/2007
// network is grid of size N

mdp

// probabilities
const double infect11;
const double infect12;
const double infect13;
const double infect21;
const double infect22;
const double infect23;
const double infect31;
const double infect32;
const double infect33;

// low nodes (those above the ceil(N/2) row)

const double detect11; 
const double detect12; 
const double detect13;   

// barrier nodes (those in the ceil(N/2) row)

const double detect21;
const double detect22;
const double detect23;

// high nodes (those below the ceil(N/2) row)

const double detect31; 
const double detect32; 
const double detect33;  

// first column (1..N)
module n11

	s11 : [0..2] init 0; // node uninfected
	// 0 - node uninfected
	// 1 - node uninfected but firewall breached
	// 2 - node infected 

	// firewall attacked (from an infected neighbour)
	[attack11_21] (s11=0) ->  detect11 : true + (1-detect11) : (s11'=1);
	[attack11_12] (s11=0) ->  detect11 : true + (1-detect11) : (s11'=1);
	// if the firewall has been breached tries to infect the node	
	[] (s11=1) -> infect11 : (s11'=2) + (1-infect11) : (s11'=0);
	// if the node is infected, then it tries to attack its neighbouring nodes
	[attack21_11] (s11=2) -> true;
	[attack12_11] (s11=2) -> true;
	
endmodule

module n21

	s21 : [0..2] init 0; // node uninfected
	// 0 - node uninfected
	// 1 - node uninfected but firewall breached
	// 2 - node infected 

	// firewall attacked (from an infected neighbour)
	[attack21_31] (s21=0) -> detect21 : true + (1-detect21) : (s21'=1);
	[attack21_22] (s21=0) -> detect21 : true + (1-detect21) : (s21'=1);
	[attack21_11] (s21=0) -> detect21 : true + (1-detect21) : (s21'=1);
	// if the firewall has been breached tries to infect the node	
	[] (s21=1) -> infect21 : (s21'=2) + (1-infect21) : (s21'=0);
	// if the node is infected, then it tries to attack its neighbouring nodes
	[attack31_21] (s21=2) -> true;
	[attack22_21] (s21=2) -> true;
	[attack11_21] (s21=2) -> true;

endmodule

module n31=n11[s11=s31,detect11=detect31,infect11=infect31,attack21_11=attack21_31,attack12_11=attack32_31,attack11_21=attack31_21,attack11_12=attack31_32] endmodule

// second column
module n12=n21[s21=s12,detect21=detect12,infect11=infect21,attack31_21=attack13_12,attack22_21=attack22_12,attack11_21=attack11_12,attack21_31=attack12_13,attack21_22=attack12_22,attack21_11=attack12_11] endmodule

module n22
	
	s22 : [0..2] init 0; // node uninfected
	// 0 - node uninfected
	// 1 - node uninfected but firewall breached
	// 2 - node infected 
	
	// firewall attacked (from an infected neighbour)
	[attack22_32] (s22=0) -> detect22 : true + (1-detect22) : (s22'=1);
	[attack22_23] (s22=0) -> detect22 : true + (1-detect22) : (s22'=1);
	[attack22_12] (s22=0) -> detect22 : true + (1-detect22) : (s22'=1);
	[attack22_21] (s22=0) -> detect22 : true + (1-detect22) : (s22'=1);
	// if the firewall has been breached tries to infect the node	
	[] (s22=1) -> infect22 : (s22'=2) + (1-infect22) : (s22'=0);
	// if the node is infected, then it tries to attack its neighbouring nodes
	[attack32_22] (s22=2) -> true;
	[attack23_22] (s22=2) -> true;
	[attack12_22] (s22=2) -> true;
	[attack21_22] (s22=2) -> true;
	
endmodule

module n32=n21[s21=s32,detect21=detect32,infect21=infect32,attack31_21=attack33_32,attack22_21=attack22_32,attack11_21=attack31_32,attack21_31=attack32_33,attack21_22=attack32_22,attack21_11=attack32_31] endmodule

// columns 3..N-1

// column N
module n13=n11[s11=s13,detect11=detect13,infect11=infect31,attack21_11=attack23_13,attack12_11=attack12_13,attack11_21=attack13_23,attack11_12=attack13_12] endmodule
module n23=n21[s21=s23,detect21=detect23,infect21=infect23,attack31_21=attack33_23,attack22_21=attack22_23,attack11_21=attack13_23,attack21_31=attack23_33,attack21_22=attack23_22,attack21_11=attack23_13] endmodule

module n33

	s33 : [0..2] init 2; // node infected;
	// 0 - node uninfected
	// 1 - node uninfected but firewall breached
	// 2 - node infected 

	// firewall attacked (from an infected neighbour)
	[attack33_32] (s33=0) ->  detect33 : true + (1-detect33) : (s33'=1);
	[attack33_23] (s33=0) ->  detect33 : true + (1-detect33) : (s33'=1);
	// if the firewall has been breached tries to infect the node	
	[] (s33=1) -> infect33 : (s33'=2) + (1-infect33) : (s33'=0);
	// if the node is infected, then it tries to attack its neighbouring nodes
	[attack32_33] (s33=2) -> true;
	[attack23_33] (s33=2) -> true;
	
endmodule

// reward structure (number of attacks)
rewards "attacks"

	// corner nodes

	[attack11_12] true : 1;
	[attack11_21] true : 1;
	[attack31_21] true : 1;
	[attack31_32] true : 1;
	[attack13_12] true : 1;
	[attack13_23] true : 1;
	[attack33_32] true : 1;
	[attack33_23] true : 1;
	
	// edge nodes

	[attack12_13] true : 1;
	[attack12_11] true : 1;
	[attack12_22] true : 1;
	[attack21_31] true : 1;
	[attack21_11] true : 1;
	[attack21_22] true : 1;
	[attack32_33] true : 1;
	[attack32_31] true : 1;
	[attack32_22] true : 1;
	[attack23_33] true : 1;
	[attack23_13] true : 1;
	[attack23_22] true : 1;

	// middle nodes
	 
	[attack22_32] true : 1;
	[attack22_23] true : 1;
	[attack22_12] true : 1;
	[attack22_21] true : 1;  

endrewards
