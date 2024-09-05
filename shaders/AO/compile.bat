glslang --target-env spirv1.5 -V -o %~dp0\..\spv\ao_rgen.spv %~dp0\ao.rgen
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\..\spv\main_pass_rchit.spv %~dp0\main_pass.rchit
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\..\spv\ao_pass_rchit.spv %~dp0\ao_pass.rchit
if errorlevel 1 exit /b %ERRORLEVEL%
glslang --target-env spirv1.5 -V -o %~dp0\..\spv\ao_rmiss.spv %~dp0\ao.rmiss
if errorlevel 1 exit /b %ERRORLEVEL%
