/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Liang Zhao and Timo Heister, Clemson University, 2016
 */

// @sect3{Include files}

// As usual, we start by including some well-known files:
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

// To transfer solutions between meshes, this file is included:
#include <deal.II/numerics/solution_transfer.h>

// This file includes UMFPACK: the direct solver:
#include <deal.II/lac/sparse_direct.h>

// And the one for ILU preconditioner:
#include <deal.II/lac/sparse_ilu.h>

// Timing
#include <chrono>
using namespace std::chrono;

#include <fstream>
#include <iostream>

namespace Step57
{
  using namespace dealii;

  // @sect3{The <code>NavierStokesProblem</code> class template}

  // This class manages the matrices and vectors described in the
  // introduction: in particular, we store a BlockVector for the current
  // solution, current Newton update, and the line search update.  We also
  // store two AffineConstraints objects: one which enforces the Dirichlet
  // boundary conditions and one that sets all boundary values to zero. The
  // first constrains the solution vector while the second constraints the
  // updates (i.e., we never update boundary values, so we force the relevant
  // update vector values to be zero).
  template <int dim>
  class StationaryNavierStokes
  {
  public:
    StationaryNavierStokes(const unsigned int degree);
    void run(const int &m,
            unsigned int &picard_iter,
            double Re,
            double &AA_time);

  private:

    void setup_dofs();

    void initialize_system();

    void assemble(const bool initial_step, const bool assemble_matrix);

    void assemble_system(const bool initial_step);

    void assemble_rhs(const bool initial_step);

    void solve(const bool initial_step);

    void refine_mesh();

    void process_solution(unsigned int refinement);

    void output_results(const unsigned int refinement_cycle) const;

    void newton_iteration(const double       tolerance,
                          const unsigned int max_n_line_searches,
                          const unsigned int max_n_refinements,
                          const bool         is_initial_step,
                          const bool         output_result);

    void picard_iteration(const double       tolerance,
                          const bool         is_initial_step,
                          const bool         output_result,
                          const int          m,
                          unsigned int       &picard_iter);

    void compute_initial_guess(double step_size,
                               unsigned int &picard_iter,
                               const int m);

    double                               viscosity;
    double                               gamma;
    const unsigned int                   degree;
    std::vector<types::global_dof_index> dofs_per_block;

    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    BlockSparsityPattern      sparsity_pattern;
    SparsityPattern           sparsity_pattern_nb;
    BlockSparseMatrix<double> system_matrix;
    //BlockSparseMatrix<double> system_norm_matrix;
    SparseMatrix<double>      system_norm_matrix;
    SparseMatrix<double>      pressure_mass_matrix;

    int AA_norm = 2; // 0 - l^2 norm (identity matrix)
                     // 1 - L^2 norm (mass matrix)
                     // 2 - H^1 norm (stiffness matrix)
    int refinement = 6;

    double AA_time_sum;

    BlockVector<double> present_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;
    BlockVector<double> evaluation_point;

    void Anderson_Acceleration(FullMatrix<double> &F,
                               FullMatrix<double> &u_tilde,
                               int &AA_iter,
                               const int &m);
  };

  // @sect3{Boundary values and right hand side}

  // In this problem we set the velocity along the upper surface of the cavity
  // to be one and zero on the other three walls. The right hand side function
  // is zero so we do not need to set the right hand side function in this
  // tutorial. The number of components of the boundary function is
  // <code>dim+1</code>. We will ultimately use
  // VectorTools::interpolate_boundary_values to set boundary values, which
  // requires the boundary value functions to have the same number of
  // components as the solution, even if all are not used. Put another way: to
  // make this function happy we define boundary values for the pressure even
  // though we will never actually use them.
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim + 1)
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));
    if (component == 0 && std::abs(p[dim - 1] - 1.0) < 1e-10)
      return 1.0;

    return 0;
  }

  // @sect3{BlockSchurPreconditioner for Navier Stokes equations}
  //
  // As discussed in the introduction, the preconditioner in Krylov iterative
  // methods is implemented as a matrix-vector product operator. In practice,
  // the Schur complement preconditioner is decomposed as a product of three
  // matrices (as presented in the first section). The $\tilde{A}^{-1}$ in the
  // first factor involves a solve for the linear system $\tilde{A}x=b$. Here
  // we solve this system via a direct solver for simplicity. The computation
  // involved in the second factor is a simple matrix-vector
  // multiplication. The Schur complement $\tilde{S}$ can be well approximated
  // by the pressure mass matrix and its inverse can be obtained through an
  // inexact solver. Because the pressure mass matrix is symmetric and
  // positive definite, we can use CG to solve the corresponding linear
  // system.
  template <class PreconditionerMp>
  class BlockSchurPreconditioner : public Subscriptor
  {
  public:
    BlockSchurPreconditioner(double                           gamma,
                             double                           viscosity,
                             const BlockSparseMatrix<double> &S,
                             const SparseMatrix<double> &     P,
                             const PreconditionerMp &         Mppreconditioner);

    void vmult(BlockVector<double> &dst, const BlockVector<double> &src) const;

  private:
    const double                     gamma;
    const double                     viscosity;
    const BlockSparseMatrix<double> &stokes_matrix;
    const SparseMatrix<double> &     pressure_mass_matrix;
    const PreconditionerMp &         mp_preconditioner;
    SparseDirectUMFPACK              A_inverse;
  };

  // We can notice that the initialization of the inverse of the matrix at the
  // top left corner is completed in the constructor. If so, every application
  // of the preconditioner then no longer requires the computation of the
  // matrix factors.

  template <class PreconditionerMp>
  BlockSchurPreconditioner<PreconditionerMp>::BlockSchurPreconditioner(
    double                           gamma,
    double                           viscosity,
    const BlockSparseMatrix<double> &S,
    const SparseMatrix<double> &     P,
    const PreconditionerMp &         Mppreconditioner)
    : gamma(gamma)
    , viscosity(viscosity)
    , stokes_matrix(S)
    , pressure_mass_matrix(P)
    , mp_preconditioner(Mppreconditioner)
  {
    A_inverse.initialize(stokes_matrix.block(0, 0));
  }

  template <class PreconditionerMp>
  void BlockSchurPreconditioner<PreconditionerMp>::vmult(
    BlockVector<double> &      dst,
    const BlockVector<double> &src) const
  {
    Vector<double> utmp(src.block(0));

    {
      SolverControl solver_control(1000, 1e-6 * src.block(1).l2_norm());
      SolverCG<Vector<double>> cg(solver_control);

      dst.block(1) = 0.0;
      cg.solve(pressure_mass_matrix,
               dst.block(1),
               src.block(1),
               mp_preconditioner);
      dst.block(1) *= -(viscosity + gamma);
    }

    {
      stokes_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;
      utmp += src.block(0);
    }

    A_inverse.vmult(dst.block(0), utmp);
  }

  // @sect3{StationaryNavierStokes class implementation}
  // @sect4{StationaryNavierStokes::StationaryNavierStokes}
  //
  // The constructor of this class looks very similar to the one in step-22. The
  // only difference is the viscosity and the Augmented Lagrangian coefficient
  // <code>gamma</code>.
  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes(const unsigned int degree)
    : viscosity(1.0 / 1.0)
    , gamma(1.0)
    , degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
  {}

  // @sect4{StationaryNavierStokes::setup_dofs}
  //
  // This function initializes the DoFHandler enumerating the degrees of freedom
  // and constraints on the current mesh.
  template <int dim>
  void StationaryNavierStokes<dim>::setup_dofs()
  {
    system_matrix.clear();
    system_norm_matrix.clear();
    pressure_mass_matrix.clear();

    // The first step is to associate DoFs with a given mesh.
    dof_handler.distribute_dofs(fe);

    // We renumber the components to have all velocity DoFs come before
    // the pressure DoFs to be able to split the solution vector in two blocks
    // which are separately accessed in the block preconditioner.
    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

    // In Newton's scheme, we first apply the boundary condition on the solution
    // obtained from the initial step. To make sure the boundary conditions
    // remain satisfied during Newton's iteration, zero boundary conditions are
    // used for the update $\delta u^k$. Therefore we set up two different
    // constraint objects.
    const FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }
    nonzero_constraints.close();

    {
      zero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }
    zero_constraints.close();

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << dof_u << " + " << dof_p << ')' << std::endl;
  }

  // @sect4{StationaryNavierStokes::initialize_system}
  //
  // On each mesh the SparsityPattern and the size of the linear system
  // are different. This function initializes them after mesh refinement.
  template <int dim>
  void StationaryNavierStokes<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DynamicSparsityPattern dsp_nb(dof_handler.n_dofs(),dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      DoFTools::make_sparsity_pattern(dof_handler, dsp_nb, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
      sparsity_pattern_nb.copy_from(dsp_nb);
    }

    system_matrix.reinit(sparsity_pattern);
    system_norm_matrix.reinit(sparsity_pattern_nb);

    present_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
  }

  // @sect4{StationaryNavierStokes::assemble}
  //
  // This function builds the system matrix and right hand side that we
  // currently work on. The @p initial_step argument is used to determine
  // which set of constraints we apply (nonzero for the initial step and zero
  // for the others). The @p assemble_matrix argument determines whether to
  // assemble the whole system or only the right hand side vector,
  // respectively.
  template <int dim>
  void StationaryNavierStokes<dim>::assemble(const bool initial_step,
                                             const bool assemble_matrix)
  {
    if (assemble_matrix)
    {
      system_matrix = 0;
      if (initial_step)
      {
        system_norm_matrix = 0;
      }
    }

    system_rhs = 0;

    QGauss<dim> quadrature_formula(degree + 2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_quadrature_points |
                              update_JxW_values | update_gradients);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_norm_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // For the linearized system, we create temporary storage for present
    // velocity and gradient, and present pressure. In practice, they are all
    // obtained through their shape functions at quadrature points.

    std::vector<Tensor<1, dim>> present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>> present_velocity_gradients(n_q_points);
    std::vector<double>         present_pressure_values(n_q_points);

    std::vector<double>         div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
    std::vector<double>         phi_p(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_norm_matrix = 0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(
          evaluation_point, present_velocity_gradients);

        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);

        // The assembly is similar to step-22. An additional term with gamma
        // as a coefficient is the Augmented Lagrangian (AL), which is
        // assembled via grad-div stabilization.  As we discussed in the
        // introduction, the bottom right block of the system matrix should be
        // zero. Since the pressure mass matrix is used while creating the
        // preconditioner, we assemble it here and then move it into a
        // separate SparseMatrix at the end (same as in step-22).
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                phi_u[k]      = fe_values[velocities].value(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (assemble_matrix)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        // Newton iteration section
/*
                        local_matrix(i, j) +=
                          (viscosity *
                             scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                           present_velocity_gradients[q] * phi_u[j] * phi_u[i] +
                           grad_phi_u[j] * present_velocity_values[q] *
                             phi_u[i] -
                           div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                           gamma * div_phi_u[j] * div_phi_u[i] +
                           phi_p[i] * phi_p[j]) *
                          fe_values.JxW(q);
*/

                          // Picard iteration section

                          local_matrix(i, j) +=
                            (viscosity *
                               scalar_product(grad_phi_u[j], grad_phi_u[i]) +
                               grad_phi_u[j] * present_velocity_values[q] *
                                 phi_u[i] -
                             div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j] +
                             gamma * div_phi_u[j] * div_phi_u[i] +
                             phi_p[i] * phi_p[j]) *
                            fe_values.JxW(q);

                          // For the norm and inner product in the AA step
                          if (initial_step)
                          {
                            if (AA_norm == 1)
                            {
                              // Mass matrix
                              local_norm_matrix(i, j) +=
                                phi_u[j] * phi_u[i] * fe_values.JxW(q);
                            }
                            else if (AA_norm == 2)
                            {
                              // Stiffness matrix
                              local_norm_matrix(i, j) +=
                                scalar_product(grad_phi_u[j], grad_phi_u[i]) *
                              fe_values.JxW(q);
                            }
                          }

                      }
                  }

                double present_velocity_divergence =
                  trace(present_velocity_gradients[q]);

                local_rhs(i) +=
                  (-viscosity * scalar_product(present_velocity_gradients[q],
                                               grad_phi_u[i]) -
                   present_velocity_gradients[q] * present_velocity_values[q] *
                     phi_u[i] +
                   present_pressure_values[q] * div_phi_u[i] +
                   present_velocity_divergence * phi_p[i] -
                   gamma * present_velocity_divergence * div_phi_u[i]) *
                  fe_values.JxW(q);

/*
                  local_rhs(i) +=
                    (-viscosity * scalar_product(present_velocity_gradients[q],
                                                 grad_phi_u[i]) +
                     present_pressure_values[q] * div_phi_u[i] +
                     present_velocity_divergence * phi_p[i] -
                     gamma * present_velocity_divergence * div_phi_u[i]) *
                    fe_values.JxW(q);
                    */
              }
          }

        cell->get_dof_indices(local_dof_indices);

        const AffineConstraints<double> &constraints_used =
          initial_step ? nonzero_constraints : zero_constraints;

        if (assemble_matrix)
          {
            constraints_used.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        system_matrix,
                                                        system_rhs);


            constraints_used.distribute_local_to_global(local_norm_matrix,
                                                        local_dof_indices,
                                                        system_norm_matrix);
          }
        else
          {
            constraints_used.distribute_local_to_global(local_rhs,
                                                        local_dof_indices,
                                                        system_rhs);
          }
      }

    if (assemble_matrix)
      {
        // Finally we move pressure mass matrix into a separate matrix:
        pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
        pressure_mass_matrix.copy_from(system_matrix.block(1, 1));

        // Note that settings this pressure block to zero is not identical to
        // not assembling anything in this block, because this operation here
        // will (incorrectly) delete diagonal entries that come in from
        // hanging node constraints for pressure DoFs. This means that our
        // whole system matrix will have rows that are completely
        // zero. Luckily, FGMRES handles these rows without any problem.
        system_matrix.block(1, 1) = 0;
      }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::assemble_system(const bool initial_step)
  {
    assemble(initial_step, true);
  }

  template <int dim>
  void StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step)
  {
    assemble(initial_step, false);
  }

  // @sect4{StationaryNavierStokes::solve}
  //
  // In this function, we use FGMRES together with the block preconditioner,
  // which is defined at the beginning of the program, to solve the linear
  // system. What we obtain at this step is the solution vector. If this is
  // the initial step, the solution vector gives us an initial guess for the
  // Navier Stokes equations. For the initial step, nonzero constraints are
  // applied in order to make sure boundary conditions are satisfied. In the
  // following steps, we will solve for the Newton update so zero
  // constraints are used.
  template <int dim>
  void StationaryNavierStokes<dim>::solve(const bool initial_step)
  {
    const AffineConstraints<double> &constraints_used =
      initial_step ? nonzero_constraints : zero_constraints;

    SolverControl solver_control(system_matrix.m(),
                                 1e-4 * system_rhs.l2_norm(),
                                 true);

    SolverFGMRES<BlockVector<double>> gmres(solver_control);
    SparseILU<double>                 pmass_preconditioner;
    pmass_preconditioner.initialize(pressure_mass_matrix,
                                    SparseILU<double>::AdditionalData());

    const BlockSchurPreconditioner<SparseILU<double>> preconditioner(
      gamma,
      viscosity,
      system_matrix,
      pressure_mass_matrix,
      pmass_preconditioner);

    // Solves system_matrix * newton_update = system_rhs...
    // WE CAN'T INCLUDE THE INITIAL VELOCITY
    gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
    std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

    constraints_used.distribute(newton_update);
  }

  // @sect4{StationaryNavierStokes::refine_mesh}
  //
  // After finding a good initial guess on the coarse mesh, we hope to
  // decrease the error through refining the mesh. Here we do adaptive
  // refinement similar to step-15 except that we use the Kelly estimator on
  // the velocity only. We also need to transfer the current solution to the
  // next mesh using the SolutionTransfer class.
  template <int dim>
  void StationaryNavierStokes<dim>::refine_mesh()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    const FEValuesExtractors::Vector velocity(0);
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      present_solution,
      estimated_error_per_cell,
      fe.component_mask(velocity));

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.0);

    triangulation.prepare_coarsening_and_refinement();
    SolutionTransfer<dim, BlockVector<double>> solution_transfer(dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(present_solution);
    triangulation.execute_coarsening_and_refinement();

    // First the DoFHandler is set up and constraints are generated. Then we
    // create a temporary BlockVector <code>tmp</code>, whose size is
    // according with the solution on the new mesh.
    setup_dofs();

    BlockVector<double> tmp(dofs_per_block);

    // Transfer solution from coarse to fine mesh and apply boundary value
    // constraints to the new transferred solution. Note that present_solution
    // is still a vector corresponding to the old mesh.
    solution_transfer.interpolate(present_solution, tmp);
    nonzero_constraints.distribute(tmp);

    // Finally set up matrix and vectors and set the present_solution to the
    // interpolated data.
    initialize_system();
    present_solution = tmp;
  }

  // @sect4{StationaryNavierStokes<dim>::newton_iteration}
  //
  // This function implements the Newton iteration with given tolerance, maximum
  // number of iterations, and the number of mesh refinements to do.
  //
  // The argument <code>is_initial_step</code> tells us whether
  // <code>setup_system</code> is necessary, and which part, system matrix or
  // right hand side vector, should be assembled. If we do a line search, the
  // right hand side is already assembled while checking the residual norm in
  // the last iteration. Therefore, we just need to assemble the system matrix
  // at the current iteration. The last argument <code>output_result</code>
  // determines whether or not graphical output should be produced.
  template <int dim>
  void StationaryNavierStokes<dim>::newton_iteration(
    const double       tolerance,
    const unsigned int max_n_line_searches,
    const unsigned int max_n_refinements,
    const bool         is_initial_step,
    const bool         output_result)
  {
    bool first_step = is_initial_step;

    for (unsigned int refinement_n = 0; refinement_n < max_n_refinements + 1;
         ++refinement_n)
      {
        unsigned int line_search_n = 0;
        double       last_res      = 1.0;
        double       current_res   = 1.0;
        std::cout << "grid refinements: " << refinement_n << std::endl
                  << "viscosity: " << viscosity << std::endl;

        while ((first_step || (current_res > tolerance)) &&
               line_search_n < max_n_line_searches)
          {
            if (first_step)
              {
                setup_dofs();
                initialize_system();
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);
                present_solution = newton_update;
                nonzero_constraints.distribute(present_solution);
                first_step       = false;
                evaluation_point = present_solution;
                assemble_rhs(first_step);
                current_res = system_rhs.l2_norm();
                std::cout << "The residual of initial guess is " << current_res
                          << std::endl;
                last_res = current_res;
              }
            else
              {
                evaluation_point = present_solution;
                assemble_system(first_step);
                solve(first_step);

                // To make sure our solution is getting close to the exact
                // solution, we let the solution be updated with a weight
                // <code>alpha</code> such that the new residual is smaller
                // than the one of last step, which is done in the following
                // loop. This is the same line search algorithm used in
                // step-15.
                for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
                  {
                    evaluation_point = present_solution;
                    evaluation_point.add(alpha, newton_update);
                    nonzero_constraints.distribute(evaluation_point);
                    assemble_rhs(first_step);
                    current_res = system_rhs.l2_norm();
                    std::cout << "  alpha: " << std::setw(10) << alpha
                              << std::setw(0) << "  residual: " << current_res
                              << std::endl;
                    if (current_res < last_res)
                      break;
                  }
                {
                  present_solution = evaluation_point;
                  std::cout << "  number of line searches: " << line_search_n
                            << "  residual: " << current_res << std::endl;
                  last_res = current_res;
                }
                ++line_search_n;
              }

            if (output_result)
              {
                output_results(max_n_line_searches * refinement_n +
                               line_search_n);

                if (current_res <= tolerance)
                  process_solution(refinement_n);
              }
          }

        if (refinement_n < max_n_refinements)
          {
            refine_mesh();
          }
      }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::picard_iteration(
    const double       tolerance,
    const bool         is_initial_step,
    const bool         output_result,
    const int          m,
    unsigned int       &picard_iter)
  {
    bool first_step = is_initial_step;

    // Number of times we look back in the Anderson Acceleration
    int AA_iter = 1;
    //unsigned int picard_iter = 0;
    double       current_res   = 1.0;
    std::cout << "viscosity: " << viscosity << std::endl;

    // This needs to be here for the time being
    setup_dofs();
    FullMatrix<double> F(dof_handler.n_dofs(),m + 1);
    FullMatrix<double> u_tilde(dof_handler.n_dofs(),m + 1);

    while ((first_step || (current_res > tolerance)) &&
           picard_iter < 200)
    {
      if (first_step)
      {
        setup_dofs();
        initialize_system();
        evaluation_point = present_solution;
        assemble_system(first_step);
        solve(first_step);
        present_solution = newton_update;
        nonzero_constraints.distribute(present_solution);
        first_step       = false;
        evaluation_point = present_solution;
        assemble_rhs(first_step);
        current_res = system_rhs.l2_norm();
        std::cout << "The residual of initial guess is " << current_res
                  << std::endl;
      }
      else
      {
        std::cout << "**************************" << std::endl;
        std::cout << "Picard Iteration: " << picard_iter << std::endl;

        // Set evaluation_point equal to the previous solution, it's
        // what is evaluated during the actual FEM.
        evaluation_point = present_solution;

        // Obvious: assembles the linear system
        assemble_system(first_step);
        solve(first_step);

        // the solve was done for newton_updates...
        // Does evaluation_point += newton_update * alpha
        evaluation_point.add(1.0, newton_update);
        //evaluation_point = newton_update;

        // Standard affine constraint distribution, don't touch
        nonzero_constraints.distribute(evaluation_point);

        // Similar to the other assembly... but with the rhs.
        assemble_rhs(first_step);


        // Here I believe we have \tilde{u}_{k+1}
        // Need to have this after the second picard iteration because
        // we're not supposed to look back at

        if (picard_iter > 0 && m != 0)
        {
          auto start = high_resolution_clock::now();

          Anderson_Acceleration(F,u_tilde,AA_iter,m);

          if (AA_iter < m + 1)
            AA_iter++;

          
          // Timer end
          auto end = high_resolution_clock::now();

          // Computational time
          auto duration_new = duration_cast<microseconds>(end - start);
          double time = duration_new.count();

          AA_time_sum += time;

          std::cout << "AA time: " << time * 1e-6
                    << " seconds" << std::endl;
        }

        //assemble_rhs(first_step);
        current_res = system_rhs.l2_norm();

        {
          present_solution = evaluation_point;
          std::cout << "residual: " << current_res << std::endl;
          std::cout << "**************************" << std::endl;
          std::cout << std::endl;
        }

        ++picard_iter;

        if (current_res > 1.0)
        {
          picard_iter = 1000;
          break;
        }
      }

      if (output_result)
      {
        output_results(picard_iter);

        if (current_res <= tolerance)
        {
          unsigned int refinement_n = 0;
          process_solution(refinement_n);
        }
      }
    }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::Anderson_Acceleration(
    FullMatrix<double> &F,
    FullMatrix<double> &u_tilde,
    int &AA_iter,
    const int &m)
  {
    std::cout << "Anderson step: " << AA_iter - 1 << std::endl;
    std::cout << "Anderson limit: " << m << std::endl;

    // This loop creates the matrix that stores all the solution data from
    // previous iterations.

    for (int j = 0; j < m; j++)
    {
      // New stuff that's hopefully faster than loops
      // for (long unsigned int i = 0; i < dof_handler.n_dofs(); i++)
      // {
      //   F(i, m - j) = F(i, m - 1 - j);
      //   u_tilde(i, m - j) = u_tilde(i, m - 1 - j);
      // }
      F.swap_col(m - j, m - 1 - j);
      u_tilde.swap_col(m - j, m - 1 - j);
    }

    for (long unsigned int i = 0; i < dof_handler.n_dofs(); i++)
    {
      // Put new computed value into first column of matrix
      F(i,0) = newton_update(i);
      u_tilde(i,0) = evaluation_point(i);
    }

    // Here we start the actual Anderson Acceleration process
    //
    // We split this into two sections. If AA_iter == 2, then we can pretty
    // easily solve for alpha using a direct inner product computation.
    // The second phase is more complicated, eplained there.
    if (AA_iter > 1)
    {
      // auto start = high_resolution_clock::now();

      Vector<double> alpha(AA_iter);

      if (AA_iter == 2)
      {
        // Here we do things for a simple m=1 case
        // for m=1, we can just set
        // alpha = -(w_{k+1} - w_k, w_k)_X / ||w_{k+1}-w_k||_X^2
        Vector<double> res_diff(dof_handler.n_dofs());
        Vector<double> res_prev(dof_handler.n_dofs());
        double numerator = 0;
        double denominator = 0;

        if (AA_norm == 0)
        {
          for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
          {
            numerator += (F(i,0) - F(i,1)) * F(i,1);
            denominator += (F(i,0) - F(i,1)) * (F(i,0) - F(i,1));
          }
        }
        else
        {
          for (unsigned int i = 0; i < dof_handler.n_dofs(); i++)
          {
            res_diff(i) = F(i,0) - F(i,1);
            res_prev(i) = F(i,1);
          }
          numerator = system_norm_matrix.matrix_scalar_product(res_diff,res_prev);
          denominator = system_norm_matrix.matrix_norm_square(res_diff);
        }

        alpha(0) = -numerator / denominator;
        alpha(1) = 1 - alpha(0);
      }
      else
      {
        // We use this method for m>1
        // Here is the slick method using Fhat
        //
        // We need to hardcode the alpha summing to 1 into the system itself,
        // so what we do is rearrange the system using the fact that
        // \alpha_m = 1 - alpha_1 - ... alpha_m-1.
        // This gives us the system:
        // Fhat * alpha_hat = -F_rhs, where
        // Fhat = [F_1 - F_m, F_2 = F_m, ..., F_m-1 - F_m]
        // alpha_hat = [alpha_1, ..., alpha_m-1]
        // F_rhs = -F_m

        int m = AA_iter - 1;

        // Some matrices/vectors that we'll need
        FullMatrix<double> sym_mat(m,m);
        Vector<double> rhs(m);
        Vector<double> temp(dof_handler.n_dofs());
        Vector<double> FtM(dof_handler.n_dofs());

        // Calculate F^T * M * F
        // Essentially does sym_mat = {<Fhat_i,Fhat_j>_M}
        //
        // As of now this is the dominating loop that differentiates
        // using a non-ell^2 matrix norm
        for (int i = 0; i < m; i++)
        {
          // This loop is only useful if we need to do a the Fhat^T * M
          // computation. So if we use the ell^2 norm, we omit this
          if (AA_norm != 0)
          {
            for (unsigned int j = 0; j < dof_handler.n_dofs(); j++)
            {
              // Stores elements in a vector for the norm matrix
              temp(j) = F(j,i) - F(j,m);
            }

            // Fhat_i^T * M
            system_norm_matrix.Tvmult(FtM,temp);
          }
          
          // Main loop
          for (unsigned int k = 0; k < dof_handler.n_dofs(); k++)
          {
            // Calculate the right hand side vector
            // Fhat_i^T * M * F_m 
            // or
            // Fhat_i^T * F_m
            if (AA_norm != 0)
              rhs(i) -= FtM(k) * F(k,m);
            else
              rhs(i) -= (F(k,i) - F(k,m)) * F(k,m);

            // Now for the sym_mat calc
            // sym_mat = F^T * M * F
            // or
            // sym_mat = F^T * F
            for (int j = 0; j < m; j++)
            {
              if (AA_norm != 0)
                sym_mat(i,j) += FtM(k) * (F(k,j) - F(k,m));
              else
                sym_mat(i,j) += (F(k,i) - F(k,m)) * (F(k,j) - F(k,m));
            }
          }
        }
        
        // This section is almost negligable time-wise
        FullMatrix<double> sym_mat_inv(m,m);
        sym_mat_inv.invert(sym_mat);

        Vector<double> alpha_temp(m);
        sym_mat_inv.vmult(alpha_temp,rhs);


        // Evaluate the implicitly imposed condition on alpha.
        double sum = 0;
        for (int i = 0; i < m; i++)
        {
          alpha(i) = alpha_temp(i);
          sum += alpha(i);
        }
        alpha(m) = 1 - sum;

        // END OF NEW STUFF       
      }

      std::cout << std::endl;
      std::cout << "Values for alpha:" << std::endl;
      for (int i = 0; i < AA_iter; i++)
      {
        std::cout << alpha(i) << std::endl;
      }
      std::cout << std::endl;

      // Here we calculate the updated solution, which we set equal to AA_sol
      // At first it is not obvious that this is a matrix-vector product,
      // but it's a sum of the u_tilde vectors and the alpha vectors, which
      // is more easily computed as a matrix-vector product with terms we
      // have already been using.
      Vector<double> AA_sol(dof_handler.n_dofs());
      u_tilde.vmult(AA_sol,alpha);

      evaluation_point = AA_sol;
    }
  }

  // @sect4{StationaryNavierStokes::compute_initial_guess}
  //
  // This function will provide us with an initial guess by using a
  // continuation method as we discussed in the introduction. The Reynolds
  // number is increased step-by-step until we reach the target value. By
  // experiment, the solution to Stokes is good enough to be the initial guess
  // of NSE with Reynolds number 1000 so we start there.  To make sure the
  // solution from previous problem is close to the next one, the step
  // size must be small enough.
  template <int dim>
  void StationaryNavierStokes<dim>::compute_initial_guess(double step_size,
                                                      unsigned int &picard_iter,
                                                      const int m)
  {
    const double target_Re = 1.0 / viscosity;

    bool is_initial_step = true;

    for (double Re = 10.0; Re < target_Re;
         Re        = std::min(Re + step_size, target_Re))
      {
        picard_iter = 0;
        viscosity = 1.0 / Re;
        std::cout << "Searching for initial guess with Re = " << Re
                  << std::endl;
        //newton_iteration(1e-12, 50, 0, is_initial_step, false);
        picard_iteration(1e-12, is_initial_step, false, m, picard_iter);
        is_initial_step = false;
      }
  }

  // @sect4{StationaryNavierStokes::output_results}
  //
  // This function is the same as in step-22 except that we choose a name
  // for the output file that also contains the Reynolds number (i.e., the
  // inverse of the viscosity in the current context).
  template <int dim>
  void StationaryNavierStokes<dim>::output_results(
    const unsigned int output_index) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();

    std::ofstream output(std::to_string(1.0 / viscosity) + "-solution-" +
                         Utilities::int_to_string(output_index, 4) + ".vtk");
    data_out.write_vtk(output);
  }

  // @sect4{StationaryNavierStokes::process_solution}
  //
  // In our test case, we do not know the analytical solution. This function
  // outputs the velocity components along $x=0.5$ and $0 \leq y \leq 1$ so they
  // can be compared with data from the literature.
  template <int dim>
  void StationaryNavierStokes<dim>::process_solution(unsigned int refinement)
  {
    std::ofstream f(std::to_string(1.0 / viscosity) + "-line-" +
                    std::to_string(refinement) + ".txt");
    f << "# y u_x u_y" << std::endl;

    Point<dim> p;
    p(0) = 0.5;
    p(1) = 0.5;

    f << std::scientific;

    for (unsigned int i = 0; i <= 100; ++i)
      {
        p(dim - 1) = i / 100.0;

        Vector<double> tmp_vector(dim + 1);
        VectorTools::point_value(dof_handler, present_solution, p, tmp_vector);
        f << p(dim - 1);

        for (int j = 0; j < dim; ++j)
          f << ' ' << tmp_vector(j);
        f << std::endl;
      }
  }

  // @sect4{StationaryNavierStokes::run}
  //
  // This is the last step of this program. In this part, we generate the grid
  // and run the other functions respectively. The max refinement can be set by
  // the argument.
  template <int dim>
  void StationaryNavierStokes<dim>::run(const int &m,
                                        unsigned int &picard_iter,
                                        double Re,
                                        double &AA_time)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(refinement);

    viscosity = 1.0 / Re;

    // If the viscosity is smaller than $1/1000$, we have to first search for an
    // initial guess via a continuation method. What we should notice is the
    // search is always on the initial mesh, that is the $8 \times 8$ mesh in
    // this program. After that, we just do the same as we did when viscosity
    // is larger than $1/1000$: run Newton's iteration, refine the mesh,
    // transfer solutions, and repeat.

    picard_iteration(1e-12, true, false, m, picard_iter);

    AA_time = AA_time_sum * 1e-6;

    std::cout << "Total AA time: " << AA_time
                  << " seconds" << std::endl;
  }
} // namespace Step57

int main()
{
  try
  {
    using namespace Step57;

    // Creating vectors to store things in and print them at the end
    std::vector<double> iterations;
    std::vector<double> time;
    std::vector<double> AA_time_vec;
    std::vector<int> Re = {100};
    std::vector<int> m = {0,1,2,10};
    double AA_time;

    // quantities we need to run the code.
    unsigned int picard_iter;


    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        picard_iter = 0;

        // Timer start
        auto start = high_resolution_clock::now();

        StationaryNavierStokes<2> flow(/* degree = */ 1);
        flow.run(m[j], picard_iter, Re[i], AA_time);

        // Timer end
        auto end = high_resolution_clock::now();

        // Computational time
        auto duration = duration_cast<microseconds>(end - start);

        // output stuff
        std::cout << "elapsed time: " << duration.count() * 1e-6
                  << " seconds" << std::endl;
        std::cout << "Total iterations: " << picard_iter - 1 << std::endl;
        time.push_back(duration.count() * 1e-6);
        iterations.push_back(picard_iter - 1);

        AA_time_vec.push_back(AA_time);
      }
    }

    std::cout << std::endl;
    std::cout << "Iterations count for: " << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "Re = " << Re[i] << ":" << std::endl;
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "  m = " << m[j] << ": "
                  << iterations[j + m.size() * i] << std::endl;
      }
    }

    std::cout << std::endl;
    std::cout << "Time (seconds) for: " << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "Re = " << Re[i] << ":" << std::endl;
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "  m = " << m[j] << ": "
                  << time[j + m.size() * i] << std::endl;
      }
    }

    std::cout << std::endl;
    std::cout << "AA Time ratio for: " << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "Re = " << Re[i] << ":" << std::endl;
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "  m = " << m[j] << ": "
                  << AA_time_vec[j + m.size() * i] / time[j + m.size() * i] << std::endl;
      }
    }

    std::cout << std::endl;
    std::cout << "iteration table" << std::endl;
    std::cout << "\\begin{tabular}{|c||c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "Iterations & $m=0$ & $m=1$ & $m=2$ & $m=10$\\\\" << std::endl;
    std::cout << "\\hline" << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "$Re = " << Re[i] << "$ ";
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "& " << iterations[j + m.size() * i] << " ";
      }
      std::cout << "\\\\" << std::endl;
      std::cout << "\\hline" << std::endl;
    }
    std::cout << "\\end{tabular}" << std::endl;

    std::cout << std::endl;
    std::cout << "Time table" << std::endl;
    std::cout << "\\begin{tabular}{|c||c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "Time & $m=0$ & $m=1$ & $m=2$ & $m=10$\\\\" << std::endl;
    std::cout << "\\hline" << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "$Re = " << Re[i] << "$ ";
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "& " << time[j + m.size() * i] << " ";
      }
      std::cout << "\\\\" << std::endl;
      std::cout << "\\hline" << std::endl;
    }
    std::cout << "\\end{tabular}" << std::endl;

    std::cout << std::endl;
    std::cout << "AA minimization time table" << std::endl;
    std::cout << "\\begin{tabular}{|c||c|c|c|c|}" << std::endl;
    std::cout << "\\hline" << std::endl;
    std::cout << "Time & $m=0$ & $m=1$ & $m=2$ & $m=10$\\\\" << std::endl;
    std::cout << "\\hline" << std::endl;
    for (long unsigned int i = 0; i < Re.size(); i++)
    {
      std::cout << "$Re = " << Re[i] << "$ ";
      for (long unsigned int j = 0; j < m.size(); j++)
      {
        std::cout << "& " << AA_time_vec[j + m.size() * i] 
          / time[j + m.size() * i] << " ";
      }
      std::cout << "\\\\" << std::endl;
      std::cout << "\\hline" << std::endl;
    }
    std::cout << "\\end{tabular}" << std::endl;

  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
