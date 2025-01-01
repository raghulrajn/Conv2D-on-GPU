/*
 * Copyright (c) 2014 Steffen Kieß
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef CORE_IMAGE_HPP_INCLUDED
#define CORE_IMAGE_HPP_INCLUDED

#include <Core/BoostFilesystem.hpp>

#include <ostream>
#include <vector>

namespace Core {
class ImageFileOptions {
 public:
  uint32_t maxValue_ = 0;
  bool oneline_ = false;

  ImageFileOptions() {}

  ImageFileOptions& maxValue(uint32_t maxValue) {
    maxValue_ = maxValue;
    return *this;
  }
  ImageFileOptions& oneline(bool oneline) {
    oneline_ = oneline;
    return *this;
  }
};

void writeImagePGM(std::ostream& stream, const uint8_t* data, size_t width,
                   size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void writeImagePGM(const Core::Path& filename, const uint8_t* data,
                   size_t width, size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void writeImagePGM(const Core::Path& filename, const std::vector<uint8_t>& data,
                   size_t width, size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void imageFloatToByte8(const std::vector<float>& input,
                       std::vector<uint8_t>& output, uint8_t maxValue);
void writeImagePGM8(const Core::Path& filename, const std::vector<float>& data,
                    size_t width, size_t height,
                    const ImageFileOptions& options = ImageFileOptions());

void writeImagePGM(std::ostream& stream, const uint16_t* data, size_t width,
                   size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void writeImagePGM(const Core::Path& filename, const uint16_t* data,
                   size_t width, size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void writeImagePGM(const Core::Path& filename,
                   const std::vector<uint16_t>& data, size_t width,
                   size_t height,
                   const ImageFileOptions& options = ImageFileOptions());
void imageFloatToByte16(const std::vector<float>& input,
                        std::vector<uint16_t>& output, uint16_t maxValue);
void writeImagePGM16(const Core::Path& filename, const std::vector<float>& data,
                     size_t width, size_t height,
                     const ImageFileOptions& options = ImageFileOptions());

void writeImagePGM(const Core::Path& filename, const std::vector<float>& data,
                   size_t width, size_t height,
                   const ImageFileOptions& options = ImageFileOptions());

void writeImagePPM(std::ostream& stream, const uint8_t* data, size_t width,
                   size_t height);
void writeImagePPM(const Core::Path& filename, const uint8_t* data,
                   size_t width, size_t height);
void writeImagePPM(const Core::Path& filename, const std::vector<uint8_t>& data,
                   size_t width, size_t height);
void writeImagePPM(std::ostream& stream, const uint16_t* data, size_t width,
                   size_t height, int bpp);
void writeImagePPM(const Core::Path& filename, const uint16_t* data,
                   size_t width, size_t height, int bpp);
void writeImagePPM(const Core::Path& filename,
                   const std::vector<uint16_t>& data, size_t width,
                   size_t height, int bpp);
void imageFloatToByteCol(const std::vector<float>& input,
                         std::vector<uint8_t>& output);
void writeImagePPM(const Core::Path& filename, const std::vector<float>& data,
                   size_t width, size_t height);

void readImagePGM(std::istream& stream, std::vector<float>& data, size_t& width,
                  size_t& height);
void readImagePGM(const Core::Path& filename, std::vector<float>& data,
                  size_t& width, size_t& height);
}  // namespace Core

#endif  // !CORE_IMAGE_HPP_INCLUDED
