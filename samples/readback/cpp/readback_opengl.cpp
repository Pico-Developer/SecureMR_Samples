//
// Created by ByteDance on 12/17/25.
//
#ifdef XR_USE_GRAPHICS_API_OPENGL_ES
#include "readback_file.h"
#include "gfxwrapper_opengl.h"

namespace SecureMR {
  void ReadbackCheck::initializeGraphicsContext() {}
  void ReadbackCheck::OutputOpenGLTextureToPath(XrReadbackTextureImageOpenGLPICO *vTexture,
                                                     const std::string &path) {
    uint32_t srcImage = vTexture->texId;
    uint32_t width = 512, height = 512;
    GLuint fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, srcImage, 0);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
      printf("Framebuffer not complete: 0x%x\n", status);
      return;
    }

    std::vector<unsigned char> pixels(width * height * 4);  // RGBA8

    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    //  for (int y = 0; y < height / 2; ++y) {
    //    int opp = height - 1 - y;
    //    for (int x = 0; x < width * 4; ++x) std::swap(pixels[y * width * 4 + x], pixels[opp * width * 4 + x]);
    //  }

    auto ret = stbi_write_png(path.c_str(), (int)width, (int)height, 4, pixels.data(), (int)width * 4);
    if (ret == 0) {
      Log::Write(Log::Level::Info, Fmt("readback stb write failed: %d", ret));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
  }

  bool ReadbackCheck::OutputReadbackTextureToPath(const XrReadbackTexturePICO &texture, const std::string &path) {
    XrReadbackTextureImageOpenGLPICO opengl_texture;

    auto ret = mReadbackController->RetrieveTexture(texture, (XrReadbackTextureImageBaseHeaderPICO*)&opengl_texture);
    if (!ret)
    {
      return false;
    }
    OutputOpenGLTextureToPath(&opengl_texture, path);
    return true;
  }
}

#endif


