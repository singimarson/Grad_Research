/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2008 - 2020 by the deal.II authors
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
 // included for set_boundary

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

// Time dependency
#include <deal.II/base/discrete_time.h>

#include <fstream>
#include <iostream>

namespace Step57
{
  using namespace dealii;


  template <int dim>
  class StationaryNavierStokes
  {
  public:
    StationaryNavierStokes(const unsigned int degree);
    void run(/* const unsigned int refinement */);

  private:
    void setup_dofs();

    void initialize_system();

    void assemble(const bool initial_step, const bool assemble_matrix, const double dt);

    void assemble_system(const bool initial_step, const double dt);

    void assemble_rhs(const bool initial_step, const double dt);

    void solve(const bool initial_step);

    void output_results(const double time) const;

    void newton_iteration(const double       tolerance,
                          const unsigned int max_n_line_searches,
                          //const unsigned int max_n_refinements,
                          const bool         is_initial_step,
                          //const bool         output_result
                          const double      dt
                          );

    void compute_initial_guess(double step_size,
                               const double dt);

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
    BlockSparseMatrix<double> system_matrix;
    SparseMatrix<double>      pressure_mass_matrix;
    // step 22 stuff
    BlockSparsityPattern      preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> preconditioner_matrix;

    //--------------

    BlockVector<double> present_solution;
    BlockVector<double> old_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;
    BlockVector<double> evaluation_point;
  };

//---------------------------------------------------------------------------

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
    if (dim == 2 && component == 0 && std::abs(p[0]) < 1e-10)
      return 1.5 *4.0 * p[1] * (0.41 - p[1]) / (.41 * .41);
    else if (dim == 3 && component == 0 && std::abs(p[1]) < 1e-10)
      return 1.5 * 4.0 * p[0] * (.41 - p[0]) * p[2] * (.41 - p[2]) / (.41 * .41);
    return 0;
  }

  //----------------------------------------------------------------------------

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

  //----------------------------------------------------------------------------------

  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes(const unsigned int degree)
    : viscosity(1.0/1000.0) // adjusted from 7500 to 100
    , gamma(0.0)
    , degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
  {}

  template <int dim>
  void StationaryNavierStokes<dim>::setup_dofs()
  {
    system_matrix.clear();
    pressure_mass_matrix.clear();

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];


    FEValuesExtractors::Vector velocities(0);
    {
      nonzero_constraints.clear();

      DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2, //boundary id
                                               BoundaryValues<dim>(),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               3, //boundary id
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

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(
                                                 dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));

      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
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

  template <int dim>
  void StationaryNavierStokes<dim>::initialize_system()
  {
    {
      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);
      DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints);
      sparsity_pattern.copy_from(dsp);
    }

    system_matrix.reinit(sparsity_pattern);

    present_solution.reinit(dofs_per_block);

    // Added for previous time step
    old_solution.reinit(dofs_per_block);
    newton_update.reinit(dofs_per_block);
    system_rhs.reinit(dofs_per_block);
  }


  template <int dim>
  void StationaryNavierStokes<dim>::assemble(const bool initial_step,
                                             const bool assemble_matrix,
                                             const double dt)
  {
    if (assemble_matrix)
      system_matrix = 0;

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
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // For the linearized system, we create temporary storage for present
    // velocity and gradient, and present pressure. In practice, they are all
    // obtained through their shape functions at quadrature points.

    std::vector<Tensor<1, dim>>             present_velocity_values(n_q_points);
    std::vector<Tensor<2, dim>>             present_velocity_gradients(n_q_points);
    std::vector<SymmetricTensor<2, dim>>    present_velocity_symgrad(n_q_points);
    std::vector<double>                     present_pressure_values(n_q_points);

    std::vector<double>                     div_phi_u(dofs_per_cell);
    std::vector<Tensor<1, dim>>             phi_u(dofs_per_cell);
    std::vector<Tensor<2, dim>>             grad_phi_u(dofs_per_cell);
    std::vector<double>                     phi_p(dofs_per_cell);
    std::vector<SymmetricTensor<2, dim>>    symgrad_phi_u(dofs_per_cell);

    // Time dependence edits
    std::vector<Tensor<1, dim>>             old_solution_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        fe_values[velocities].get_function_values(evaluation_point,
                                                  present_velocity_values);

        fe_values[velocities].get_function_gradients(evaluation_point,
                                                     present_velocity_gradients);

        fe_values[pressure].get_function_values(evaluation_point,
                                                present_pressure_values);

        // Time dependence
        fe_values[velocities].get_function_values(old_solution,
                                                  old_solution_values);

        fe_values[velocities].get_function_symmetric_gradients(evaluation_point,
                                                               present_velocity_symgrad);

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                symgrad_phi_u[k] = fe_values[velocities].symmetric_gradient(k, q);
                phi_u[k]      = fe_values[velocities].value(k, q);
                phi_p[k]      = fe_values[pressure].value(k, q);
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                double present_velocity_divergence =
                  trace(present_velocity_gradients[q]);

                if (assemble_matrix)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        local_matrix(i, j) +=
                          (viscosity * scalar_product(grad_phi_u[j], grad_phi_u[i])
                           //------------------------------------------------------
                           // EMAC terms
                           + 2 * present_velocity_symgrad[q] * phi_u[j] * phi_u[i]          // (2D(u_k)u_{k+1},v)
                           + 2 * symgrad_phi_u[j] * present_velocity_values[q] * phi_u[i]   // (2D(u_{k+1})u_k,v)

                           + div_phi_u[j] * present_velocity_values[q] * phi_u[i]  // ((div u_{k+1})u_k,v)
                           + present_velocity_divergence * phi_u[j] * phi_u[i]     // ((div u_k)u_{k+1},v)
                           //------------------------------------------------------
                           - div_phi_u[i] * phi_p[j]
                           - phi_p[i] * div_phi_u[j]
                           + gamma * div_phi_u[j] * div_phi_u[i]

                           + phi_p[i] * phi_p[j]
                           // Time dependent component
                           + (1.0 / dt) * phi_u[j] * phi_u[i]
                           )
                           * fe_values.JxW(q);
                      }
                  }

                local_rhs(i) +=
                  (- viscosity * scalar_product(present_velocity_gradients[q],grad_phi_u[i])
                   //------------------------------------------------------------------------
                   // EMAC terms
                   - 2 * present_velocity_symgrad[q] * present_velocity_values[q] * phi_u[i]      // (2D(u_k)u_k,v)
                   - present_velocity_divergence * present_velocity_values[q] * phi_u[i]          // ((div u_k)u_k,v)
                   //--------------------------------------------------------------------------
                   + present_pressure_values[q] * div_phi_u[i]
                   + present_velocity_divergence * phi_p[i]
                   - gamma * present_velocity_divergence * div_phi_u[i]
                   // time dependent terms
                   + (1.0 / dt) * old_solution_values[q] * phi_u[i]
                   - (1.0 / dt) * present_velocity_values[q] * phi_u[i]
                   )
                   * fe_values.JxW(q);
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

        system_matrix.block(1, 1) = 0;
      }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::assemble_system(const bool initial_step, const double dt)
  {
    assemble(initial_step, true, dt);
  }

  template <int dim>
  void StationaryNavierStokes<dim>::assemble_rhs(const bool initial_step, const double dt)
  {
    assemble(initial_step, false, dt);
  }


  template <int dim>
  void StationaryNavierStokes<dim>::solve(const bool initial_step)
  {
    const AffineConstraints<double> &constraints_used =
      initial_step ? nonzero_constraints : zero_constraints;

  if (dim == 2)
  {
    SparseDirectUMFPACK solver;
    solver.initialize(system_matrix);
    solver.vmult(newton_update, system_rhs);
    constraints_used.distribute(newton_update);
  }
  else if (dim == 3)
  {
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

      gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
      std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

      constraints_used.distribute(newton_update);
  }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::newton_iteration(
    const double       tolerance,
    const unsigned int max_n_line_searches,
    const bool         is_initial_step,
    const double       dt)
  {
    bool first_step = is_initial_step;

    unsigned int line_search_n = 0;
    double       last_res      = 1.0;
    double       current_res   = 1.0;

    while ((first_step || (current_res > tolerance)) &&
           line_search_n < max_n_line_searches)
      {
        if (first_step)
          {
            evaluation_point = present_solution;
            assemble_system(first_step, dt);
            solve(first_step);
            present_solution = newton_update;
            nonzero_constraints.distribute(present_solution);
            first_step       = false;
            evaluation_point = present_solution;
            assemble_rhs(first_step, dt);
            current_res = system_rhs.l2_norm();
            std::cout << "The residual of initial guess is " << current_res
                      << std::endl;
            last_res = current_res;

          }
        else
          {
            evaluation_point = present_solution;
            assemble_system(first_step, dt);
            solve(first_step);
\
            for (double alpha = 1.0; alpha > 1e-5; alpha *= 0.5)
              {
                evaluation_point = present_solution;
                evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(evaluation_point);
                assemble_rhs(first_step, dt);
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
      }
  }


  template <int dim>
  void StationaryNavierStokes<dim>::output_results(
    const double time) const
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

    std::string filename = std::to_string(1.0 / viscosity) + "-solution1-" +
        Utilities::int_to_string(time, 2) + ".vtk";
    std::cout << "writing " << filename << std::endl;
    std::ofstream output(filename);


    data_out.write_vtk(output);
  }


  template <int dim>
  void StationaryNavierStokes<dim>::run(/* const unsigned int refinement */)
  {
    GridGenerator::channel_with_cylinder(triangulation, .03, 1, 0.0, true);
    triangulation.refine_global(2);

    setup_dofs();
    initialize_system();

    old_solution = 0.;
    // Time dependence stuff
    double dt = 0.01;
    int index = 0;
    output_results(index);

    DiscreteTime time(0., 8., dt);

    std::cout << std::endl;
    std::cout << "current time is: " << time.get_current_time() << std::endl;
    std::cout << std::endl;

    newton_iteration(1e-12, 50, true, dt);

    output_results(index);

    old_solution = present_solution;

    while(time.is_at_end() == false)
    {
        time.advance_time();

        double t = time.get_current_time();

        std::cout << std::endl;
        std::cout << "current time is: " << t << std::endl;
        std::cout << std::endl;

        newton_iteration(1e-12, 50, false, dt);

        old_solution = present_solution;

        //if (std::fmod(t,1) < 1e-5)
        {
            ++index;
            output_results(index);
        }
    }
  }
} // namespace Step57

int main()
{
  try
    {
      using namespace Step57;

      StationaryNavierStokes<2> flow(/* degree = */ 1);
      flow.run();
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
