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

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <chrono>
using namespace std::chrono;

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

    void picard_iteration(const double       tolerance,
                          const bool         is_initial_step,
                          const int          m,
                          unsigned int       &picard_iter);

    void Anderson_Acceleration(FullMatrix<double> &F_temp,
                               FullMatrix<double> &u_tilde_temp,
                               int &AA_iter,
                               const int &m);

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
    SparseMatrix<double>      system_norm_matrix;
    SparseMatrix<double>      pressure_mass_matrix;

    int AA_norm = 0; // 0 - l^2 norm (identity matrix)
                     // 1 - L^2 norm (mass matrix)
                     // 2 - H^1 norm (stiffness matrix)
    int refinement = 6;

    double AA_time_sum;

    BlockVector<double> present_solution;
    BlockVector<double> newton_update;
    BlockVector<double> system_rhs;
    BlockVector<double> evaluation_point;
  };

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

  template <int dim>
  StationaryNavierStokes<dim>::StationaryNavierStokes(const unsigned int degree)
    : viscosity(1.0 / 1.0)
    , gamma(1.0)
    , degree(degree)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , fe(FE_Q<dim>(degree + 1), dim, FE_Q<dim>(degree), 1)
    , dof_handler(triangulation)
  {}

  template <int dim>
  void StationaryNavierStokes<dim>::setup_dofs()
  {
    system_matrix.clear();
    system_norm_matrix.clear();
    pressure_mass_matrix.clear();

    dof_handler.distribute_dofs(fe);

    std::vector<unsigned int> block_component(dim + 1, 0);
    block_component[dim] = 1;
    DoFRenumbering::component_wise(dof_handler, block_component);

    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
    unsigned int dof_u = dofs_per_block[0];
    unsigned int dof_p = dofs_per_block[1];

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
        pressure_mass_matrix.reinit(sparsity_pattern.block(1, 1));
        pressure_mass_matrix.copy_from(system_matrix.block(1, 1));

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

    gmres.solve(system_matrix, newton_update, system_rhs, preconditioner);
    std::cout << "FGMRES steps: " << solver_control.last_step() << std::endl;

    constraints_used.distribute(newton_update);
  }

  template <int dim>
  void StationaryNavierStokes<dim>::picard_iteration(
    const double       tolerance,
    const bool         is_initial_step,
    const int          m,
    unsigned int       &picard_iter)
  {
    bool first_step = is_initial_step;

    int AA_iter = 1;
    AA_time_sum = 0;
    double       current_res   = 1.0;

    setup_dofs();
    FullMatrix<double> F_temp(dof_handler.n_dofs(),m + 1);
    FullMatrix<double> u_tilde_temp(dof_handler.n_dofs(),m + 1);

    while ((first_step || (current_res > tolerance)) &&
           picard_iter < 250)
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
      }
      else
      {
        evaluation_point = present_solution;
        assemble_system(first_step);
        solve(first_step);
        evaluation_point.add(1.0, newton_update);
        nonzero_constraints.distribute(evaluation_point);
        assemble_rhs(first_step);

        if (picard_iter > 0 && m != 0)
        {
          Anderson_Acceleration(F_temp,u_tilde_temp,AA_iter,m);

          if (AA_iter < m + 1)
            AA_iter++;
        }
        current_res = system_rhs.l2_norm();

        {
          present_solution = evaluation_point;
        }

        ++picard_iter;

        if (current_res > 1.0)
        {
          picard_iter = 1000;
          break;
        }
      }
    }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::Anderson_Acceleration(
    FullMatrix<double> &F_temp,
    FullMatrix<double> &u_tilde_temp,
    int &AA_iter,
    const int &m)
  {
    std::cout << "Anderson step: " << AA_iter - 1 << std::endl;
    std::cout << "Anderson limit: " << m << std::endl;

    // Create submatrices for F_temp and u_tilde_temp when k < m
    FullMatrix<double> F(dof_handler.n_dofs(),AA_iter);
    FullMatrix<double> u_tilde(dof_handler.n_dofs(),AA_iter);
    for (int j = 0; j < m; j++)
    {
      F_temp.swap_col(m - j, m - 1 - j);
      u_tilde_temp.swap_col(m - j, m - 1 - j);
    }

    for (long unsigned int i = 0; i < dof_handler.n_dofs(); i++)
    {
      // Put new computed value into first column of matrix
      F_temp(i,0) = newton_update(i);
      u_tilde_temp(i,0) = evaluation_point(i);

      // Setting up the submatrix
      for (int j = 0; j < AA_iter; j++)
      {
        F(i,j) = F_temp(i,j);
        u_tilde(i,j) = u_tilde_temp(i,j);
      }
    }

    // We split this into two sections. If AA_iter == 2, then we can pretty
    // easily solve for alpha using a direct inner product computation.
    // The second phase is more complicated, eplained there.
    if (AA_iter > 1)
    {
      Vector<double> alpha(AA_iter);

      if (AA_iter == 2)
      {
        // Here we do things for a simple m=1 case
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
        // We need to hardcode the alpha summing to 1 into the system itself,
        // so what we do is rearrange the system using the fact that
        // \alpha_m = 1 - alpha_1 - ... alpha_m-1.
        // This gives us the system:
        // Fhat * alpha_hat = -F_rhs, where
        // Fhat = [F_1 - F_m, F_2 = F_m, ..., F_m-1 - F_m]
        // alpha_hat = [alpha_1, ..., alpha_m-1]
        // F_rhs = -F_m

        int m = AA_iter - 1;

        // Calculate F^T * M * F
        // Essentially does sym_mat = {<Fhat_i,Fhat_j>_M}
        FullMatrix<double> sym_mat(m,m);
        Vector<double> rhs(m);
        Vector<double> temp(dof_handler.n_dofs());
        Vector<double> FtM(dof_handler.n_dofs());
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
            // Fhat_i^T * M * F_m (or Fhat_i^T * F_m)
            if (AA_norm != 0)
              rhs(i) -= FtM(k) * F(k,m);
            else
              rhs(i) -= (F(k,i) - F(k,m)) * F(k,m);

            // Now for the sym_mat calc
            // sym_mat = F^T * M * F (or sym_mat = F^T * F)
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

        Vector<double> alpha_new(m);
        sym_mat_inv.vmult(alpha_new,rhs);


        // Evaluate the implicitly imposed condition on alpha.
        double sum = 0;
        for (int i = 0; i < m; i++)
        {
          alpha(i) = alpha_new(i);
          sum += alpha(i);
        }
        alpha(m) = 1 - sum;    
      }

      // Here we calculate the updated solution, which we set equal to AA_sol
      // We cannot use evaluation_point as the input for vmult because it
      // is a BlockVector.
      Vector<double> AA_sol(dof_handler.n_dofs());
      u_tilde.vmult(AA_sol,alpha);
      evaluation_point = AA_sol;
    }
  }

  template <int dim>
  void StationaryNavierStokes<dim>::run(const int &m,
                                        unsigned int &picard_iter,
                                        double Re,
                                        double &AA_time)
  {
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(refinement);

    viscosity = 1.0 / Re;

    picard_iteration(1e-12, true, m, picard_iter);

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
    std::vector<int> Re = {2500};
    std::vector<int> m = {10};
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

        std::cout << "elapsed time: " << duration.count() * 1e-6
                  << " seconds" << std::endl;
        std::cout << "Total iterations: " << picard_iter - 1 << std::endl;
        time.push_back(duration.count() * 1e-6);
        iterations.push_back(picard_iter - 1);

        AA_time_vec.push_back(AA_time);
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
    std::cout << "AA time ratio table" << std::endl;
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
