#include <OpenANN/util/EigenWrapper.h>
#include <cstring>


namespace OpenANN
{
    namespace util
    {
        Eigen::MatrixXd fromStdVector(const std::vector<std::vector<double>>& x)
        {
            Eigen::MatrixXd out(x.size(), x[0].size());
            for(std::size_t r = 0; r < x.size(); r++)
            {
                for(std::size_t c = 0; c < x[r].size(); c++)
                {
                    out(r,c) = x[r][c];
                }
            }

            return out;
        }

        std::vector<std::vector<double>> toStdVector(const Eigen::MatrixXd& x)
        {
            std::vector<std::vector<double>> res(x.rows());
            for(std::size_t r = 0; r < res.size(); r++)
            {
                res[r].reserve(x.cols());
                for(std::size_t c = 0; c < (std::size_t)x.cols(); c++)
                {
                    res[r].push_back(x(r,c));
                }
            }

            return res;
        }
    }
}

void pack(Eigen::VectorXd& vec, int components, ...)
{
  std::va_list pointers;
  va_start(pointers, components);

  int offset = 0;
  for(int n = 0; n < components; n++)
  {
    int size = va_arg(pointers, int);
    void* p = va_arg(pointers, void*);
    std::memcpy(vec.data()+offset, p, size*sizeof(double));
    offset += size;
  }

  va_end(pointers);
}

void unpack(const Eigen::VectorXd& vec, int components, ...)
{
  std::va_list pointers;
  va_start(pointers, components);

  int offset = 0;
  for(int n = 0; n < components; n++)
  {
    int size = va_arg(pointers, int);
    void* p = va_arg(pointers, void*);
    std::memcpy(p, vec.data()+offset, size*sizeof(double));
    offset += size;
  }

  va_end(pointers);
}
