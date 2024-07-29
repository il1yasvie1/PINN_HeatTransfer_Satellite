# PINN_HeatTransfer_Satellite
solving heat transfer equation using physical informed neural network
# Solving Space Thermal Supervising System using Physical Informed Neural Network

---

Author: Sanyang Liu

Date: 29/7/2024

## Table of Contents

1. Project Overview

2. Mathematics Background for the Heat Conduction Problems

3. Introduction to Physics-Informed Neural Networks

4. Data Simulation in COMSOL

5. Solving Transient Problem with Solar Heat Source

---

## 1. Project Overview

Predicting the temperature distribution on a satellite is critical for ensuring its proper functionality. Many of the satellite's electronic devices require operation within a specific temperature range to maintain optimal performance. When exposed to solar radiation following its emergence from Earth's shadow, the sun-facing side of the satellite can experience a substantial rise in temperature. Without effective thermal management actions, this can affect the performance of the satellite's internal instruments. This study simplifies the satellite model to a unit cube and employs a non-supervised Physics-Informed Neural Network (PINN) to predict the temperature distribution across the satellite. The predictions made by the PINN are validated against simulation data obtained from COMSOL, providing a comparative analysis to assess the accuracy of the proposed approach.

## 2. Mathematics Background for the Heat Conduction Problems

Define the domain of the problem as $\Omega_\mathbf{x} = [0,1]^3$, the time domain as $\Omega_t =[0,\infty)$. The surface exposed to the solar heat source is defined as $\partial\Omega_{x=1}=\{ (1, y, z):y, z\in[0,1]\}$. For every position $(x, y, z)\in \Omega_\mathbf{x}$ and time $t\in \Omega_t$, we aim to get the predictions of the temperature $T(x, y, z, t)$ satisfying the following partial differential equation:

$$
\begin{align*}
\rho C_p \partial_t T = \nabla_\mathbf{x} \cdot (k\nabla_\mathbf{x} T) 
, \quad & (\mathbf{x}, t) \in \Omega_{\mathbf{x}} \times \Omega_t \\
n \cdot (k\nabla_\mathbf{x} T) = \epsilon \sigma (T_{amb}^4 - T^4) 
, \quad & (\mathbf{x}, t)\in 
(\partial \Omega_\mathbf{x} \setminus \partial \Omega_{x=1}) \times \Omega_t\\
n \cdot (k\nabla_\mathbf{x} T) = Q_b
, \quad & (\mathbf{x}, t)\in \partial \Omega_{x=1} \times \Omega_t\\
T(\mathbf{x}, 0)=T_0 ,\quad & \mathbf{x}\in \Omega_\mathbf{x}
\end{align*}
$$

The parameters in the equation are listed in the following table:

| Symbol     | Meaning                   | Value               |
|:----------:|:------------------------- |:-------------------:|
| $\rho $    | density                   | 1                   |
| $C_p$      | heat capacity             | 1                   |
| $k$        | thermal conductivity      | 167                 |
| $\epsilon$ | emissivity                | 0.1                 |
| $\sigma$   | Stefan-Boltzmann Constant | $5.67\times10^{-8}$ |
| $T_{amb}$  | ambient temperature       | 3                   |
| $T_0$      | initial temperature       | 300                 |

## 3. Introduction to Physics-Informed Neural Networks

## 4.  Data Simulation in COMSOL

## 5. Solving Transient Problem with Solar Heat Source
