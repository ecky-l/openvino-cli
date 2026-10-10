
- Don't rush, and ask clarifying questions.
- Don't use git without explicit permission.
- Evaluate the runtime environment by calling "eval $(./setupenv.sh)" before every cmake build and run (once for every subshell).
- Keep code comments short, but understandable.
- Never use new/delete or even malloc/free. Instead do always use RAII, as far as possible.
- Prefer references over pointers. When pointers are needed, prefer std smart pointers (std::shared_ptr etc.).
- Use const wherever possible.
