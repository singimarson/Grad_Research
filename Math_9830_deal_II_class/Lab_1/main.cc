#include <iostream>
#include <fstream>
#include <vector>

/**
 * A simple vector class the represents a vector of R^n.
 */
class Vector
{
public:
  std::vector<double> values;

  explicit Vector(size_t size = 0)
  {
    values.resize(size);
  }


  void load(const char *filename)
  {
    std::ifstream f(filename);
    values.clear();
    while (!f.eof())
      {
        double value;
        f >> value;
        values.push_back(value);
      }
  }

  void print()
  {
    for (int i=0; i<static_cast<int>(values.size()); ++i)
      std::cout << values[i] << '\n';
    std::cout << std::flush;
  }
};


/**
 * An n by n sparse matrix
 */
class SparseMatrix
{
public:
  /**
   * Compute the matrix-vector product: dest = A*src
   */
  void mat_vec(Vector &dest, const Vector &src)
  {
    for (int i=0; i<n; ++i)
      {
        dest.values[i]=0.0;
        for (int idx=row_start[i]; idx<row_start[i+1]; ++idx)
          dest.values[i] += values[idx]*src.values[column_indices[idx]];
      }
  }

  /**
   * Compute the transposed matrix-vector product: dest = A'*src
   */
  void mat_vec_transposed(Vector &dest, const Vector &src)
  {
      for (int i=0; i<n; ++i)
      {
          dest.values[i]=0.0; // Need to declare these guys as zero before doing anything
      }
      for (int i=0; i<n; ++i)
        {
          for (int idx=row_start[i]; idx<row_start[i+1]; ++idx)
              // All we do here is switch up the clumn_indices, which is intuitive because of what the transpose is
            dest.values[column_indices[idx]] += values[idx]*src.values[i];
        }
  }

  void load(const char *filename)
  {
    std::ifstream f(filename);
    values.clear();
    column_indices.clear();
    row_start.clear();

    int lastrow = -1;
    int row, col;
    double value;
    while (!f.eof())
      {
        f >> row >> col >> value;
        if (lastrow != row)
          {
            row_start.push_back(values.size());
            lastrow = row;
          }
        if (col!=-1)
          {
            values.push_back(value);
            column_indices.push_back(col);
          }

      }
    // point after last row:
    row_start.push_back(values.size());
    n = row_start.size()-1;
  }

  void print()
  {
    for (int r=0; r<n; ++r)
      {
        for (int idx=row_start[r]; idx<row_start[r+1]; ++idx)
          std::cout << "(" << r << "," << column_indices[idx] << "): "
                    << values[idx] << std::endl;
      }
  }

  void print_full()
  {
    // As of now this is pumping out an extra 5.5, gotta fix it
    for (int r=0; r<n; ++r)
      {
        int idx = row_start[r];
        for (int c=0; c<n; ++c)
          {
            if (c > column_indices[idx] && idx < row_start[r+1]-1) // It was iterating idx too much
                // check was implemented to make sure idx didn't go above it's iteration limit of row_start[r+1]-1
              ++idx;
            if (c == column_indices[idx])
              std::cout << values[idx] << "\t";
            else
              std::cout << "-\t";
          }
        std::cout << std::endl;
      }
  }

  std::vector<double> values;
  std::vector<int> column_indices;
  std::vector<int> row_start;
  int n;
};



int main()
{
  Vector x;
  SparseMatrix mat;

  x.load("vector");
  std::cout << "x:" << std::endl;
  x.print();

  mat.load("matrix");
  std::cout << "A:" << std::endl;
  mat.print();
  std::cout << "A:" << std::endl;
  mat.print_full();

  std::cout << std::endl << "A*x" << std::endl;
  Vector y(4);
  mat.mat_vec(y, x);
  y.print();

  std::cout << std::endl << "A'*x" << std::endl;
  Vector z(4);
  mat.mat_vec_transposed(z, x);
  z.print();
}
